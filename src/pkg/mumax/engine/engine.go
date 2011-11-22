//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package engine

// Author: Arne Vansteenkiste

import (
	. "mumax/common"
	"fmt"
	"strings"
)

// The global simulation engine
var engine Engine

// Returns the global simulation engine
func GetEngine() *Engine {
	return &engine
}

// Engine is the heart of a multiphysics simulation.
// The engine stores named quantities like "m", "B", "alpha", ...
// An acyclic graph structure consisting of interconnected quantities
// determines what should be calculated and when.
type Engine struct {
	size3D_        [3]int            // INTENRAL
	size3D         []int             // size of the FD grid, nil means not yet set
	cellSize_      [3]float64        // INTENRAL
	cellSize       []float64         // size of the FD cells, nil means not yet set
	quantity       map[string]*Quant // maps quantity names onto their data structures
	solver         []Solver          // each solver does the time stepping for its own quantities
	time           *Quant            // time quantity is always present
	dt             *Quant            // time step quantity is always present
	timer          Timer             // For benchmarking
	modules        []Module          // loaded modules 
	crontabs       map[int]Notifier  // periodical jobs, indexed by handle
	outputTables   map[string]*Table // open output table files, indexed by file name
	_outputID      int               // index for output numbering
	_lastOutputT   float64           // time of last output ID increment
	_handleCount   int               // used to generate unique handle IDs for various object passed out
	outputDir      string            // output directory
	filenameFormat string            // Printf format string for file name numbering. Must consume one integer.
}

// Gets an ID number to identify the current time. Used to number output files. E.g. the 7 in "m000007.omf". Files with the same OutputID correspond to the same simulation time. 
func (e *Engine) OutputID() int {
	t := e.time.Scalar()
	if t != e._lastOutputT {
		e._lastOutputT = t
		e._outputID++
	}
	return e._outputID
}

// Returns ++_handleCount. Used to identify objects like crontabs so they can later by manipulated through this ID.
func (e *Engine) NewHandle() int {
	e._handleCount++ // Let's not use 0 as a valid handle.
	return e._handleCount
}

// Left-hand side and right-hand side indices for Engine.ode[i]
const (
	LHS = 0
	RHS = 1
)

// Initializes the global simulation engine
func Init() {
	(&engine).init()
}

// initialize
func (e *Engine) init() {
	e.quantity = make(map[string]*Quant)
	// special quantities time and dt are always present
	e.AddQuant("t", SCALAR, VALUE, Unit("s"))
	e.AddQuant("dt", SCALAR, VALUE, Unit("s"))
	e.time = e.Quant("t")
	e.dt = e.Quant("dt")
	e.dt.SetVerifier(Positive)
	e.modules = make([]Module, 0)
	e.crontabs = make(map[int]Notifier)
	e.outputTables = make(map[string]*Table)
	e.filenameFormat = "%06d"
	e.timer.Start()
}

// Shuts down the engine. Closes all open files, etc.
func (e *Engine) Close() {
	for _, t := range e.outputTables {
		t.Close()
	}
}

//__________________________________________________________________ set/get

// Sets the FD grid size
func (e *Engine) SetGridSize(size3D []int) {
	Debug("Engine.SetGridSize", size3D)
	Assert(len(size3D) == 3)
	if e.size3D == nil {
		e.size3D = e.size3D_[:]
		copy(e.size3D, size3D)
	} else {
		panic(InputErr("Grid size already set"))
	}
}

// Gets the FD grid size
func (e *Engine) GridSize() []int {
	if e.size3D == nil {
		panic(InputErr("Grid size should be set first"))
	}
	return e.size3D
}

// Sets the FD cell size
func (e *Engine) SetCellSize(size []float64) {
	Debug("Engine.SetCellSize", size)
	Assert(len(size) == 3)
	if e.cellSize == nil {
		e.cellSize = e.cellSize_[:]
		copy(e.cellSize, size)
	} else {
		panic(InputErr("Cell size already set"))
	}
}

// Gets the FD cell size
func (e *Engine) CellSize() []float64 {
	if e.cellSize == nil {
		panic(InputErr("Cell size should be set first"))
	}
	return e.cellSize
}

// Gets the total number of FD cells
func (e *Engine) NCell() int {
	return e.size3D_[0] * e.size3D_[1] * e.size3D_[2]
}

// Retrieve a quantity by its name.
// Lookup is case-independent
func (e *Engine) Quant(name string) *Quant {
	lname := strings.ToLower(name)
	if q, ok := e.quantity[lname]; ok {
		return q
	} else {
		e.addDerivedQuant(name)
		if q, ok := e.quantity[lname]; ok {
			return q
		} else {
			panic(Bug("engine.Quant()"))
		}
	}
	return nil //silence gc
}

// Derived quantities are averages, components, etc. of existing quantities.
// They are added to the engine on-demand.
// Syntax:
//	"<q>"  : average of q
//	"q.x"  : x-component of q, must be vector
//	"q.xx" : xx-component of q, must be tensor
//	"<q.x>": average of x-component of q.
func (e *Engine) addDerivedQuant(name string) {
	// average
	if strings.HasPrefix(name, "<") && strings.HasSuffix(name, ">") {
		origname := name[1 : len(name)-1]
		original := e.Quant(origname)

		e.AddQuant(name, original.nComp, VALUE, original.unit)
		derived := e.Quant(name)
		e.Depends(name, origname)
		derived.updater = NewAverageUpdater(original, derived)
		return
	}
	// component
	if strings.Contains(name, ".") {
		split := strings.Split(name, ".")
		if len(split) != 2 {
			panic(InputErr("engine: undefined quantity: " + name))
		}
		origname, compname := split[0], strings.ToLower(split[1])
		orig := e.Quant(origname)

		// parse component string ("X" -> 0)
		comp := -1
		ok := false
		switch orig.nComp {
		default:
			panic(InputErr(orig.Name() + " has no component " + compname))
		case 3:
			comp, ok = VectorIndex[strings.ToUpper(compname)]
			comp = 2 - comp // userspace
		case 6:
			comp, ok = TensorIndex[strings.ToUpper(compname)]
			if comp < 3 {
				comp = 2 - comp
			} // userspace
			if comp == YZ {
				comp = XY
			}
			if comp == XY {
				comp = YZ
			}
		}
		if !ok {
			panic(InputErr("invalid component:" + compname))
		}

		derived := orig.Component(comp)
		derived.name = orig.name + "." + strings.ToLower(compname) // hack, graphviz can't handle "."
		e.addQuant(derived)
		e.Depends(derived.name, origname)
		return
	}
	panic(InputErr("engine: undefined quantity: " + name))
}

//__________________________________________________________________ add

// Returns true if the named module is already loaded.
func (e *Engine) HasModule(name string) bool {
	for _, m := range e.modules {
		if m.Name() == name {
			return true
		}
	}
	return false
}

// Low-level module load, not aware of dependencies
func (e *Engine) LoadModule(name string) {
	if e.HasModule(name) {
		return
	}
	module := GetModule(name)
	Log("Loaded module", module.Name(), ":", module.Description())
	module.Load(e)
	e.modules = append(e.modules, module)
}

// Add an arbitrary quantity. Name tag is case-independent.
func (e *Engine) AddQuant(name string, nComp int, kind QuantKind, unit Unit, desc ...string) {
	e.addQuant(newQuant(name, nComp, e.size3D, kind, unit, desc...))
}

func (e *Engine) addQuant(q *Quant) {
	lname := strings.ToLower(q.name)

	// quantity should not yet be defined
	if _, ok := e.quantity[lname]; ok {
		panic(Bug("engine: Already defined: " + q.name))
	}

	e.quantity[lname] = q
}

// AddQuant(name, nComp, VALUE)
func (e *Engine) AddValue(name string, nComp int, unit Unit) {
	e.AddQuant(name, nComp, VALUE, unit)
}

// AddQuant(name, nComp, FIELD)
func (e *Engine) AddField(name string, nComp int, unit Unit) {
	e.AddQuant(name, nComp, FIELD, unit)
}

// AddQuant(name, nComp, MASK)
func (e *Engine) AddMask(name string, nComp int, unit Unit) {
	e.AddQuant(name, nComp, MASK, unit)
}

// Mark childQuantity to depend on parentQuantity.
// Multiply adding the same dependency has no effect.
func (e *Engine) Depends(childQuantity string, parentQuantities ...string) {
	child := e.Quant(childQuantity)
	for _, parentQuantity := range parentQuantities {
		parent := e.Quant(parentQuantity)

		for _, p := range child.parents {
			if p.name == parentQuantity {
				return // Dependency is already defined, do not add it twice
				//panic(Bug("Engine.addDependency(" + childQuantity + ", " + parentQuantity + "): already present"))
			}
		}

		child.parents[parent.Name()] = parent
		parent.children[child.Name()] = child
	}
}

// Add a 1st order differential equation:
//	d y / d t = diff
// E.g.: ODE1("m", "torque")
// No direct dependency should be declared between the arguments.
func (e *Engine) ODE1(y, diff string) {
	yQ := e.Quant(y)
	dQ := e.Quant(diff)

	// check that two solvers are not trying to update the same output quantity
	if e.solver != nil {
		for _, solver := range e.solver {
			_, out := solver.Deps()
			for _, q := range out {
				if q.Name() == y {
					panic(Bug("Already in ODE: " + y))
				}
			}
		}
	}
	e.solver = append(e.solver, NewEuler(e, yQ, dQ)) // TODO: choose solver type here
}

//________________________________________________________________________________ step

// Takes one ODE step.
// It is the solver's responsibility to Update/Invalidate its dependencies as needed.
func (e *Engine) Step() {
	if len(e.solver) == 0 {
		panic(InputErr("engine.Step: no differential equations loaded."))
	}

	// step, but hide result in buffer so we don't interfere with other solvers depending on the result
	for _, solver := range e.solver {
		solver.AdvanceBuffer()
	}
	// now that all solvers have updated behind the screens, we can make the result visible.
	for _, solver := range e.solver {
		solver.CopyBuffer()
	}

	// advance time
	e.time.SetScalar(e.time.Scalar() + e.dt.Scalar())
	//e.time.Invalidate() // automatically

	// check if output needs to be saved
	e.notifyAll()
}
//__________________________________________________________________ output

// Notifies all crontabs that a step has been taken.
func (e *Engine) notifyAll() {
	for _, tab := range e.crontabs {
		tab.Notify(e)
	}
}

// Saves the quantity once in the specified format and file name
func (e *Engine) Save(q *Quant, format string, options []string, filename string) {
	checkKinds(q, MASK, FIELD)
	out := OpenWRONLY(filename)
	defer out.Close()
	bufout := Buffer(out)
	defer bufout.Flush()
	GetOutputFormat(format).Write(bufout, q, options)
}

// Saves the quantity periodically.
func (e *Engine) AutoSave(quant string, format string, options []string, period float64) (handle int) {
	checkKinds(e.Quant(quant), MASK, FIELD)
	handle = e.NewHandle()
	e.crontabs[handle] = &AutoSave{quant, format, options, period, 0}
	Log("Auto-save", quant, "every", period, "s", "(handle ", handle, ")")
	return handle
}

func (e *Engine) Tabulate(quants []string, filename string) {
	if _, ok := e.outputTables[filename]; !ok { // table not yet open
		e.outputTables[filename] = NewTable(filename)
	}
	table := e.outputTables[filename]
	table.Tabulate(quants)
}

func (e *Engine) AutoTabulate(quants []string, filename string, period float64) (handle int) {
	for _, q := range quants {
		checkKinds(e.Quant(q), MASK, VALUE)
	}
	handle = e.NewHandle()
	e.crontabs[handle] = &AutoTabulate{quants, filename, period, 0}
	Log("Auto-tabulate", quants, "every", period, "s", "(handle ", handle, ")")
	return handle
}

// Generates an automatic file name for the quantity, given the output format.
// E.g., "dir.out/m000007.omf"
// see: outputDir, filenameFormat
func (e *Engine) AutoFilename(quant, format string) string {
	filenum := fmt.Sprintf(e.filenameFormat, e.OutputID())
	filename := quant + filenum + "." + GetOutputFormat(format).Name()
	dir := ""
	if e.outputDir != "" {
		dir = e.outputDir + "/"
	}
	return dir + filename
}

// Looks for an object with the handle number and removes it.
// Currently only looks in the crontabs.
func (e *Engine) RemoveHandle(handle int) {
	found := false
	if _, ok := e.crontabs[handle]; ok {
		e.crontabs[handle] = nil, false
		found = true
	}
	// TODO: if handles are used by other objects than crontabs, find them here
	if !found {
		Log(e.crontabs)
		panic(IOErr(fmt.Sprint("handle does not exist:", handle)))
	}
}

// INTERNAL: Used by frontend to set the output dir
func (e *Engine) SetOutputDirectory(dir string) {
	e.outputDir = dir
}

// String representation
func (e *Engine) String() string {
	str := "engine\n"
	quants := e.quantity
	for k, v := range quants {
		str += "\t" + k + "("
		for _, p := range v.parents {
			str += p.name + " "
		}
		str += ")\n"
	}
	//str += "ODEs:\n"
	//for _, ode := range e.ode {
	//	str += "d " + ode[0].Name() + " / d t = " + ode[1].Name() + "\n"
	//}
	return str
}

// DEBUG: statistics
func (e *Engine) Stats() string {
	str := fmt.Sprintln("engine running", e.timer.Seconds(), "s")
	quants := e.quantity
	for _, v := range quants {
		str += fmt.Sprintln(fill(v.Name()), "\t",
			valid(v.upToDate), " upd:", fill(v.updates),
			" inv:", fill(v.invalidates),
			valid(v.bufUpToDate), " xfer:", fill(v.bufXfers),
			" ", fmt.Sprintf("%f", v.timer.Average()*1000), "ms/upd ",
			v.multiplier, v.unit)
	}
	return str
}

func valid(b bool) string {
	if b {
		return "✓"
	}
	return "✗"
}

func fill(s interface{}) string {
	str := fmt.Sprint(s)
	for len(str) < 6 {
		str += " "
	}
	return str
}
