//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any
//  copyright notices and prominently state that you modified it, giving a relevant date.

package engine

// The methods in this file will be automatically exposed in the mumax API
// through the apigen program.
//
//	NOTE: Here the user input (X,Y,Z) is changed to internal input (Z,Y,X)

import (
	"fmt"
	"math"
	. "mumax/common"
	"mumax/gpu"
	"mumax/host"
	"os"
	"path"
	"reflect"
	"runtime"
)

const (
	FINE_T     = 100  // subdivision of time interval in dispersion calculation
	B_EXT      = 1e-3 // the amplitude of excitation used in dispersion calculation
	OUT_FORMAT = "dump"
)

// The API methods are accessible to the end-user through scripting languages.
type API struct {
	Engine *Engine
}

//________________________________________________________________________________ init

// Set the grid size.
// WARNING: convert to ZYX
func (a API) SetGridSize(x, y, z int) {
	a.Engine.SetGridSize([]int{z, y, x}) // convert to internal axes
}

// Get the grid size.
// WARNING: convert to ZYX
func (a API) GetGridSize() (x, y, z int) {
	size := a.Engine.GridSize()
	return size[Z], size[Y], size[X] // convert to internal axes
}

// Set the cell size.
// WARNING: convert to ZYX
func (a API) SetCellSize(x, y, z float64) {
	a.Engine.SetCellSize([]float64{z, y, x}) // convert to internal axes and units
}

// Get the cell size.
// WARNING: convert to ZYX, internal units
func (a API) GetCellSize() (x, y, z float64) {
	size := a.Engine.CellSize()
	return size[Z], size[Y], size[X] // convert to internal axes
}

// Get the toal size, in meters, of the simulated world.
func (a API) GetWorldSize() (x, y, z float64) {
	size := a.Engine.WorldSize()
	return size[Z], size[Y], size[X] // convert to internal axes
}

// Set periodic boundary conditions in each direction.
// A value of 0 means no periodicity in that direction (the default).
// A nonzero value means the system is infinitely reproduced in that direction.
// The magnitude of the nonzero value is a hint of how accurately the
// infinite character should be approached, if applicable.
// E.g.: for the ferromagnetic exchange interaction,
// any nonzero value will give the same result: perfect infinite periodicity.
// But for the magnetostatic interaction, the magnitude of the nonzero value
// may be used as a hint where to cut off the magnetic field.
func (a API) SetPeriodic(x, y, z int) {
	a.Engine.SetPeriodic([]int{z, y, x})
}

// Get the periodicity
// WARNING: convert to ZYX, internal units
func (a API) GetPeriodic() (x, y, z int) {
	p := a.Engine.Periodic()
	return p[Z], p[Y], p[X] // convert to internal axes
}

// Load a physics module.
func (a API) Load(name string) {
	a.Engine.LoadModule(name)
}

// Load a physics module with user-defined variables.
func (a API) LoadArgs(name string, ins, deps, outs []string) {
	a.Engine.LoadModuleArgs(name, ins, deps, outs)
}

//________________________________________________________________________________ run

// Take one solver step
func (a API) Step() {
	a.Engine.Step()
}

// Takes N solver steps
func (a API) Steps(N int) {
	a.Engine.Steps(N)
}

// Runs for a duration given in seconds.
// TODO: precise stopping time
func (a API) Run(duration float64) {
	a.Engine.Run(duration)
}

// Runs the simulation until quantity a < value
func (a API) Run_Until_Smaller(quantity string, value float64) {
	e := a.Engine
	q := e.Quant(quantity)
	Log("Running until", q.Name(), "<", value, q.Unit())
	for q.Scalar() >= value {
		e.Step()
		e.updateDash()
	}
	DashExit()
}

// Runs the simulation until quantity a > quantity b
func (a API) Run_Until_Larger(quantity string, value float64) {
	e := a.Engine
	q := e.Quant(quantity)
	Log("Running until", q.Name(), ">", value, q.Unit())
	for q.Scalar() <= value {
		e.Step()
		e.updateDash()
	}
	DashExit()
}

//________________________________________________________________________________ set quantities

// Set value of a quantity. The quantity must be of type VALUE or MASK.
// If the quantity is a MASK, the value will be multiplied by a space-dependent mask
// which typically contains dimensionless numbers between 0 and 1.
func (a API) SetV(quantity string, value []float64) {
	q := a.Engine.Quant(quantity)
	SwapXYZ(value)
	q.SetValue(value)
}

func (a API) SetValue(quantity string, value []float64) {
	Warn("setvalue deprecated: use setv")
	a.SetV(quantity, value)
}

// Used to set a quantity as a function of time. Usage:
//	SetPointwise("Quant", time0, [value0])
//	SetPointwise("Quant", time1, [value1])
//	SetPointwise("Quant", time2, [value2])
//	...
// Will make the quantity vary as a function of time, using
// piecewise linear interpolation between the defined time-value pairs.
// It is a good idea to end with something like:
//	SetPointwise("Quant", 9999, [0])
// to define the value as zero for time = infinity (after a pulse, e.g.),
// because the function has to be defined during the entire simulation.
func (a API) SetPointwise(quantity string, time float64, value []float64) {
	e := a.Engine
	q := e.Quant(quantity)
	checkKinds(q, VALUE, MASK)

	u := q.GetUpdater()
	if u == nil {
		u = newPointwiseUpdater(q)
		q.SetUpdater(u)
	}

	pointwise, ok := u.(*PointwiseUpdater)
	if !ok {
		panic(InputErrF("Can not set time-dependent", quantity, ", it is already determined in an other way:", reflect.TypeOf(u)))
	}

	SwapXYZ(value)
	pointwise.Append(time, value) // swap!

}

func (a API) SetPointwiseOf(argument string, quantity string, arg float64, value []float64) {
	e := a.Engine

	argum := e.Quant(argument)
	q := e.Quant(quantity)

	checkKinds(argum, VALUE, SCALAR)
	checkKinds(q, VALUE, MASK)

	u := q.GetUpdater()
	if u == nil {
		u = newPointwiseOfUpdater(argum, q)
		q.SetUpdater(u)
	}

	pointwise, ok := u.(*PointwiseOfUpdater)
	if !ok {
		panic(InputErrF("Can not set time-dependent", quantity, ", it is already determined in an other way:", reflect.TypeOf(u)))
	}

	SwapXYZ(value)
	pointwise.Append(arg, value) // swap!

}

// Set scalar. Convenience method for SetValue() with only one number.
// REDUNDANT?
func (a API) SetS(quantity string, value float64) {
	q := a.Engine.Quant(quantity)
	q.SetValue([]float64{value})
}

func (a API) SetScalar(quantity string, value float64) {
	Warn("setscalar deprecated: use sets or setv")
	a.SetS(quantity, value)
}

// Sets a space-dependent multiplier mask for the quantity.
// The value of the quantity (set by SetValue), will be multiplied
// by the mask value in each point of space. The mask is dimensionless
// and typically contains values between 0 and 1.
func (a API) SetMask(quantity string, mask *host.Array) {
	q := a.Engine.Quant(quantity)
	qArray := q.Array()
	if !EqualSize(mask.Size3D, qArray.Size3D()) {
		Log("Auto-resampling ", q.Name(), "from", Size(mask.Size3D), "to", Size(qArray.Size3D()))
		mask = Resample(mask, qArray.Size3D())
	}
	q.SetMask(mask)
}

// Like SetMask but reads the mask from a file.
func (a API) SetMask_File(quantity string, filename string) {
	a.SetMask(quantity, ReadFile(filename))
}

// Sets a space-dependent field quantity, like the magnetization.
func (a API) SetArray(quantity string, field *host.Array) {
	q := a.Engine.Quant(quantity)
	qArray := q.Array()
	if !EqualSize(field.Size3D, qArray.Size3D()) {
		Log("Auto-resampling ", quantity, "from", Size(field.Size3D), "to", Size(qArray.Size3D()))
		field = Resample(field, qArray.Size3D())
	}
	// setting a field when there is a non-1 multiplier is too confusing to allow
	for _, m := range q.multiplier {
		if m != 1 {
			panic(InputErr(fmt.Sprint(q.Name(), " is not an oridinary array, but has a mask + multiplier value. Did you mean to set the mask or the multiplier instead of the array?")))
		}
	}
	q.SetField(field)
}

// Like SetArray but reads the array from a file.
func (a API) SetArray_File(quantity string, filename string) {
	a.SetArray(quantity, ReadFile(filename))
}

//________________________________________________________________________________ get quantities

// Get the value of a space-independent or masked quantity.
// Returns an array with vector components or an
// array with just one element in case of a scalar quantity.
func (a API) GetV(quantity string) []float64 {
	q := a.Engine.Quant(quantity)
	q.Update() //!
	value := make([]float64, len(q.multiplier))
	copy(value, q.multiplier)
	SwapXYZ(value)
	return value
}

// DEPRECATED: same as getv()
func (a API) GetValue(quantity string) []float64 {
	return a.GetV(quantity)
}

// DEBUG: Does not update.
func (a API) DebugV(quantity string) []float64 {
	q := a.Engine.Quant(quantity)
	//q.Update() //!
	value := make([]float64, len(q.multiplier))
	copy(value, q.multiplier)
	SwapXYZ(value)
	return value
}

// Gets the quantities unit.
func (a API) Unit(quantity string) string {
	return string(a.Engine.Quant(quantity).unit)
}

// Get the value of a scalar, space-independent quantity.
// Similar to GetValue, but returns a single number.
func (a API) GetS(quantity string) float64 {
	q := a.Engine.Quant(quantity)
	q.Update() //!
	return q.Scalar()
}

// DEPRECATED: same as gets()
func (a API) GetScalar(quantity string) float64 {
	return a.GetS(quantity)
}

// Gets a space-dependent quantity. If the quantity uses a mask,
// the result is equal to GetMask() * GetValue()
func (a API) GetArray(quantity string) *host.Array {
	q := a.Engine.Quant(quantity)
	checkKinds(q, MASK, FIELD)
	q.Update() //!
	return q.Buffer()
}

// DEBUG: does not update
func (a API) DebugField(quantity string) *host.Array {
	q := a.Engine.Quant(quantity)
	checkKinds(q, MASK, FIELD)
	//q.Update() //!
	buffer := q.Buffer()
	return buffer
}

// FOR DEBUG ONLY.
// Gets the quantity's array, raw.
func (a API) Debug_GetArray(quant string) *host.Array {
	q := a.Engine.Quant(quant)
	q.Update() //!
	array := q.Array()
	buffer := q.Buffer()
	array.CopyToHost(buffer)
	return buffer
}

// Gets the value of the quantity at cell position x,y,z
func (a API) GetCell(quant string, x, y, z int) []float64 {
	q := a.Engine.Quant(quant)
	q.Update() //!
	value := make([]float64, q.NComp())
	if q.Array().IsNil() {
		for c := range value {
			value[c] = q.multiplier[c]
		}
	} else {
		for c := range value {
			value[c] = q.multiplier[c] * float64(q.Array().Get(c, z, y, x))
		}
	}
	SwapXYZ(value)
	return value
}

// Sets the value of the quantity at cell position x,y,z
func (a API) SetCell(quant string, x, y, z int, value []float64) {
	q := a.Engine.Quant(quant)
	SwapXYZ(value)
	pointers := q.Array().DevicePtr()
	if pointers[0] == 0 {
		panic(InputErr(fmt.Sprint("You are attempting to call setcell() before array for qunatity is allocated. Please consider setmask() instead.")))
	}
	//q.assureAlloc()
	for c := range value {
		q.Array().Set(c, z, y, x, float32(value[c]))
	}
	q.Invalidate() //!
}

// Sets scalar quantity uniform on each region
// @param quant (string) name of the scalar quantity to set
// @param initValues ([]float32) array containing the initial values to set. The index of each value must correpond to the concerned region.
// @note A wrapper should be defined to allow the user to give a dictionary where keys are the names of the regions.
func (a API) SetScalarUniformRegion(quant string, initValues []float32) {
	q := a.Engine.Quant(quant)
	if q.nComp != 1 {
		panic(InputErr(fmt.Sprint(q.Name(), " is not a scalar. It has ", q.nComp, "component(s).")))
	}
	Log("Set uniformly scalar field", quant)
	q.assureAlloc()
	qArray := q.Array()
	regionArray := a.Engine.Quant("regionDefinition").Array()
	gpu.InitScalarQuantUniformRegion(initValues, qArray, regionArray)
	q.Invalidate()
}

// Sets vector quantity uniform on each region
// @param quant (string) name of the scalar quantity to set
// @param initValues ([]float32) array containing the initial values to set. The index of each value must correpond to the concerned region.
// @note A wrapper should be defined to allow the user to give a dictionary where keys are the names of the regions.
func (a API) SetVectorUniformRegion(quant string, initValuesX, initValuesY, initValuesZ []float32) {
	q := a.Engine.Quant(quant)
	if q.nComp != 3 {
		panic(InputErr(fmt.Sprint(q.Name(), " is not a vector. It has ", q.nComp, "component(s).")))
	}
	if len(initValuesX) != len(initValuesY) || len(initValuesY) != len(initValuesZ) || len(initValuesX) != len(initValuesZ) {
		panic(InputErr(fmt.Sprint("Initial values are corrupted. The number of X, Y and Z components is not the same.")))
	}
	Log("Set uniformly vector field", quant)
	q.assureAlloc()
	regions := a.Engine.Quant("regionDefinition")
	gpu.InitVectorQuantUniformRegion(q.Array(), regions.Array(), initValuesX, initValuesY, initValuesZ)
	q.Invalidate()
}

// Sets vector quantity to vortex on selected regions
// @param quant (string) name of the scalar quantity to set
// @param regionsToProceed ([]bool) index correspond to region index and value is true if the region should be set to vortex. Else it is set to false.
// @param center ([]float32) array containing the coordinates of the center of the vortex
// @param axis ([]float32) array containing the coordinates of the axis of the vortex
// @param cellsize ([]float32) array containing the cell size along each axis X, Y, Z
// @param polarity (int) integer equal to +1 if the polarity is up (relatively to the axis) and -1 if the polarity is down.
// @param chirality (int) integer equal to +1 if the chirality is CCW and -1 if the chirality is CW (when the vortex is seen from the top, relatively to the axis).
// @param maxRadius (float) float reprensenting the maximum radius around the axis, that should be processed. 0 means limitless.
// @note A wrapper should be defined to allow the user to give a dictionary where keys are the names of the regions.
func (a API) SetVectorVortexRegion(quant string, regionsToProceed, center, axis, cellsize []float32, polarity, chirality int, maxRadius float32) {
	q := a.Engine.Quant(quant)
	if q.nComp != 3 {
		panic(InputErr(fmt.Sprint(q.Name(), " is not a vector. It has ", q.nComp, "component(s).")))
	}
	if polarity != -1 && polarity != 1 {
		panic(InputErr(fmt.Sprint("Polarity should be either 1 (up) or -1 (down).")))
	}
	if chirality != -1 && chirality != 1 {
		panic(InputErr(fmt.Sprint("Chirality should be either 1 (CCW) or -1 (CW).")))
	}
	if len(center) != 3 {
		panic(InputErr(fmt.Sprint("Center should have 3D coordinates instead of ", len(center), "D.")))
	}
	if len(axis) != 3 {
		panic(InputErr(fmt.Sprint("Axis should have 3D coordinates instead of ", len(axis), "D.")))
	}
	if len(cellsize) != 3 {
		panic(InputErr(fmt.Sprint("Cellsize should have 3D coordinates instead of ", len(cellsize), "D.")))
	}
	Log("Set", quant, "to vortex state")
	q.assureAlloc()
	regions := a.Engine.Quant("regionDefinition")
	regionP := []bool{}
	for c := range regionsToProceed {
		if regionsToProceed[c] == 0.0 {
			regionP = append(regionP, false)
		} else {
			regionP = append(regionP, true)
		}
	}
	//gpu.InitVectorQuantVortexRegion(q.Array(), regions.Array(), regionsToProceed, center, axis, cellsize, polarity, chirality, maxRadius)
	gpu.InitVectorQuantVortexRegion(q.Array(), regions.Array(), regionP, center, axis, cellsize, polarity, chirality, maxRadius)
	q.Invalidate()
}

// Sets scalar quantity random on selected regions. Random value are uniformly distributed between min and max value
// @param quant (string) name of the scalar quantity to set
// @param regionsToProceed ([]bool) index correspond to region index and value is true if the region should be set to vortex. Else it is set to false.
// @param max (float) upper limit of the range of random number
// @param min (float) lower limit of the range of random number
func (a API) SetScalarQuantRandomUniformRegion(quant string, regionsToProceed []float32, max, min float32) {
	q := a.Engine.Quant(quant)
	if q.nComp != 1 {
		panic(InputErr(fmt.Sprint(q.Name(), " is not a scalar. It has ", q.nComp, "components.")))
	}
	Log("Set uniformly random scalar quant", quant)
	q.assureAlloc()
	regions := a.Engine.Quant("regionDefinition")
	regionP := []bool{}
	for c := range regionsToProceed {
		if regionsToProceed[c] == 0.0 {
			regionP = append(regionP, false)
		} else {
			regionP = append(regionP, true)
		}
	}
	gpu.InitScalarQuantRandomUniformRegion(q.Array(), regions.Array(), regionP, max, min)
	q.Invalidate()
}

// Sets vector quantity random on selected regions. Random value are uniformly distributed between min and max value
// @param quant (string) name of the scalar quantity to set
// @param regionsToProceed ([]bool) index correspond to region index and value is true if the region should be set to vortex. Else it is set to false.
func (a API) SetVectorQuantRandomUniformRegion(quant string, regionsToProceed []float32) {
	q := a.Engine.Quant(quant)
	if q.nComp != 3 {
		panic(InputErr(fmt.Sprint(q.Name(), " is not a vector. It has ", q.nComp, "component(s).")))
	}
	Log("Set uniformly random vector quant", quant)
	q.assureAlloc()
	regions := a.Engine.Quant("regionDefinition")
	regionP := []bool{}
	for c := range regionsToProceed {
		if regionsToProceed[c] == 0.0 {
			regionP = append(regionP, false)
		} else {
			regionP = append(regionP, true)
		}
	}
	gpu.InitVectorQuantRandomUniformRegion(q.Array(), regions.Array(), regionP)
	q.Invalidate()
}

// ________________________________________________________________________________ save quantities

// Saves a space-dependent quantity, once. Uses the specified format and gives an automatic file name (like "m000001.png").
// See also: Save()
func (a API) Save(quantity string, format string, options []string) {
	quant := a.Engine.Quant(quantity)
	filename := a.Engine.AutoFilename(quantity, format)
	a.Engine.SaveAs(quant, format, options, filename)
}

// Saves a space-dependent quantity, once. Uses the specified format and file name.
func (a API) SaveAs(quantity string, format string, options []string, filename string) {
	a.Engine.SaveAs(a.Engine.Quant(quantity), format, options, filename)
}

// Saves a space-dependent quantity periodically, every period (expressed in seconds).
// Output appears in the output directory with automatically generated file names.
// E.g., for a quantity named "m", and format "txt" the generated files will be:
//	m00000.txt m00001.txt m00002.txt...
// See FilenameFormat() for setting the number of zeros.
// Returns an integer handle that can be used to manipulate the auto-save entry.
// E.g. remove(handle) stops auto-saving it.
// @see filenumberfomat
func (a API) AutoSave(quantity string, format string, options []string, period float64) (handle int) {
	return a.Engine.AutoSave(quantity, format, options, period)
}

// Saves a space-dependent quantity periodically to a single file, every period (expressed in seconds).
// Output appears in the output directory with automatically generated file name.
// E.g., for a quantity named "m", and format "txt" the generated files will be:
// m.dump, b.dump ...
// See FilenameFormat() for setting the number of zeros.
// This function outputs to single file by appending new data to the end of the file.
// Therefore it is meaningfull only for mumax2's 'dump' binary format.
// Returns an integer handle that can be used to manipulate the auto-save entry.
// E.g. remove(handle) stops auto-saving it.
// @see filenumberfomat
func (a API) AutoSaveSingleFile(quantity string, format string, options []string, period float64) (handle int) {
	return a.Engine.AutoSaveSingleFile(quantity, format, options, period)
}

// Saves these space-independent quantities, once.
// Their values are appended to the file, on one line.
func (a API) Tabulate(quantities []string, filename string) {
	a.Engine.Tabulate(quantities, filename)
}

// Saves any number of space-independent quantities periodically,
// every period (expressed in seconds).
// The values are appended to the file.
// Returns an integer handle that can be used to manipulate the auto-save entry.
// E.g. remove(handle) stops auto-saving it.
func (a API) AutoTabulate(quantities []string, filename string, period float64) (handle int) {
	return a.Engine.AutoTabulate(quantities, filename, period)
}

// Removes the object with given handle.
// E.g.:
//	handle = autosave(...)
//	remove(handle) # stops auto-saving
func (a API) Remove(handle int) {
	a.Engine.RemoveHandle(handle)
}

// Sets a global C-style printf format string used to generate file names for automatically saved files.
// The default "%06d" generates, e.g., "m000001.txt". "%d" would generate, e.g., "m1.txt".
func (a API) FileNumberFormat(format string) {
	a.Engine.filenameFormat = format
	Log("Using", format, "to number automatically saved files.")
}

// Returns the output directory for the running simulation.
func (a API) OutputDirectory() string {
	return a.Engine.outputDir
}

//________________________________________________________________________________ add quantities

// Add a new quantity to the multi-physics engine, its
// value is added to the (existing) sumQuantity.
// E.g.: Add_To("H", "H_1") adds a new external field
// H_1 that will be added to H.
func (a API) Add_To(sumQuantity, newQuantity string) {
	a.Add_To_Weighted(sumQuantity, newQuantity, 1.0)
}

// Add a new quantity to the multi-physics engine, its
// value is multiplied by the weight value and add to the (existing) sumQuantity.
// E.g.: Add_To_Weight("H", "H_1") adds a new external field
// H_1 that will be added to H.

func (a API) Add_To_Weighted(sumQuantity, newQuantity string, weight float64) {

	e := a.Engine
	sumQuant := e.Quant(sumQuantity)
	sumUpd, ok := sumQuant.GetUpdater().(*SumUpdater)
	if !ok {
		panic(InputErrF("Add_To: quantity ", sumQuant.Name(), " is not of type 'sum', nothing can be added to it."))
	}

	var term *Quant

	if e.HasQuant(newQuantity) {
		term = e.Quant(newQuantity)
	} else {
		term = e.AddNewQuant(newQuantity, sumQuant.NComp(), MASK, sumQuant.Unit())
	}

	sumUpd.MAddParent(term.Name(), weight)
	Log("Added new quantity", term.FullName(), "to", sumQuant.Name())

	//e := a.Engine
	//sumQuant := e.Quant(sumQuantity)
	//term := e.AddNewQuant(newQuantity, sumQuant.NComp(), MASK, sumQuant.Unit())
	//AddTermToQuant(sumQuant, term)
}

// untested
//func (a API) Add_Dot(dotQuantity, inputQuant1, inputQuant2 string) {
//	e := a.Engine
//	inQuant1 := e.Quant(inputQuant1)
//	inQuant2 := e.Quant(inputQuant2)
//	unit := "(" + inQuant1.Unit() + ")*(" + inQuant2.Unit() + ")"
//	dotQuant := e.AddNewQuant(dotQuantity, SCALAR, VALUE, unit)
//	dotQuant.SetUpdater(NewSDotUpdater(dotQuant, inQuant1, inQuant2, 1))
//	Log("Added new dot product:", dotQuantity, "=", inputQuant1, "·", inputQuant2)
//}

// Add a new quantity to the multi-physics engine, its
// value is the maximum of the absolute value of inputQuantity.
// E.g.: New_MaxAbs("max_torque", "torque") adds a new quantity
// "max_torque" whose value is max(abs(torque)). For vector
// quantities, the maximum is taken over all components.
func (a API) New_MaxAbs(newQuantity, inputQuantity string) {
	e := a.Engine
	In := e.Quant(inputQuantity)
	checkKind(In, FIELD)
	New := e.AddNewQuant(newQuantity, SCALAR, VALUE, In.Unit())
	New.SetUpdater(NewMaxAbsUpdater(In, New)) // also sets dependency
}

// Add a new quantity to the multi-physics engine, its
// value is the maximum norm of inputQuantity (a 3-component vector).
// E.g.: New_MaxNorm("maxnorm_torque", "torque") adds a new quantity
// "maxnorm_torque" whose value is max(norm(torque)).
func (a API) New_MaxNorm(newQuantity, inputQuantity string) {
	e := a.Engine
	In := e.Quant(inputQuantity)
	checkKind(In, FIELD)
	checkComp(In, 3)
	New := e.AddNewQuant(newQuantity, SCALAR, VALUE, In.Unit())
	New.SetUpdater(NewMaxNormUpdater(In, New)) // also sets dependency
}

func (a API) New_Peak(newQuantity, inputQuantity string) {
	e := a.Engine
	In := e.Quant(inputQuantity)
	checkKind(In, VALUE)
	checkComp(In, 1)
	New := e.AddNewQuant(newQuantity, SCALAR, VALUE, In.Unit())
	New.SetUpdater(NewPeakUpdater(In, New))
}

//________________________________________________________________________________ misc

// Saves an image file of the physics graph using the given file name.
// The extension determines the output format. E.g.: .png, .svg, ...
// A file with a .dot extension will be written as well.
// Rendering requires the package "graphviz" to be installed.
func (a API) SaveGraph(file string) {

	file = a.Engine.Relative(file)
	dotfile := ReplaceExt(file, ".dot")

	f, err := os.Create(dotfile)
	defer f.Close()
	CheckIO(err)
	a.Engine.WriteDot(f)
	Log("Wrote", dotfile)
	RunDot(dotfile, path.Ext(file)[1:]) // rm .
}

// DEBUG
func (a API) PrintStats() {
	Log(a.Engine.Stats())
}

// DEBUG: manually update the quantity state
func (a API) Debug_Update(quantity string) {
	a.Engine.Quant(quantity).Update()
}

// DEBUG: manually update the quantity state
func (a API) Debug_Invalidate(quantity string) {
	a.Engine.Quant(quantity).Invalidate()
}

// DEBUG: removes the updater of this quantity
func (a API) Debug_DisableUpdate(quantity string) {
	a.Engine.Quant(quantity).updater = nil
}

// DEBUG: verify all quanties' values
func (a API) Debug_VerifyAll() {
	e := a.Engine
	for _, q := range e.quantity {
		q.Verify()
	}
}

func (a API) Debug_GC() {
	Log("GC")
	runtime.GC()
}

// DEBUG: start a timer with a given identifier tag
func (a API) StartTimer(tag string) {
	EnableTimers(true)
	Start(tag)
}

// DEBUG: stop a timer with a given identifier tag.
// It must be started first.
func (a API) StopTimer(tag string) {
	Stop(tag)
}

// DEBUG: Gets the time, in seconds, recorded by the timer with this tag.
func (a API) GetTime(tag string) float64 {
	return GetTime(tag)
}

// DEBUG: Resets the timer with this tag.
func (a API) ResetTimer(tag string) {
	ResetTimer(tag)
}

// DEBUG: echos a string, can be used for synchronous output
func (a API) Echo(str string) {
	Log(str)
}

// Should be called at the end of each input file to make
// sure the input script does not terminate too early.
func (a API) Sync() {
	// The simple act of calling a function (any function)
	// causes the front-end RPC to synchronize the scripting
	// language and mumax.
	Log("Syncing")
}

// DEBUG: reads an array from a file.
func (a API) ReadFile(filename string) *host.Array {
	return ReadFile(filename)
}

// Returns the output ID corresponding to the current simulation time.
// All automatic output uses this number to identify the time corresponding
// to the saved quantity.
func (a API) OutputID() int {
	return a.Engine.OutputID()
}

//~ @ This file implements dispersion calculation API for mumax2 core
//~ @ It is intendant to replace semargl functionality
//~ @author: Mykola 'godsic' Dvornik
//~ @param fmin (float) specifies lower bound of the scan range
//~ @param fmax (float) specifies upper bound of the scan range
//~ @param steps (int)  specifies number of points within scan range
//~ @param format (int) specifies output format: 0 - Amplitude/Phase, 1 - X/Y

func (a API) GetDispersion(fmin, fmax float64, steps, format int) {
	if fmax < fmin {
		panic(InputErr("The bandwidth is negative."))
	}

	//~ calculate frequency increment
	df := (fmax - fmin) / float64(steps)

	meshSize := a.Engine.GridSize()
	cellSize := a.Engine.CellSize()
	worldSize := a.Engine.WorldSize()
	logicSize := a.Engine.PaddedSize()

	//~ calculate half BZ boundaries
	bw := make([]float64, len(cellSize))
	for i := range cellSize {
		bw[i] = 0.5 * math.Pi / cellSize[i]
	}

	qM := a.Engine.Quant("m")
	m := qM.Array()
	COMP := qM.NComp()

	//~ Save initial state
	m0 := gpu.NewArray(COMP, meshSize)
	m0.CopyFromDevice(m)
	B0 := a.GetValue("B_ext")

	//~ create spatial mask for excitation field
	//~ to precisely control spatial bandwidth of the excitation the sinc function is used
	//~ with the cutoff tuned to the boundary of the Brillouin zone of the discreet mesh

	Debug("Preparing excitation mask...")
	bMask := host.NewArray(COMP, meshSize)
	Debug(bw)

	for k := 0; k < meshSize[0]; k++ {
		z := float64(k)*cellSize[0] - 0.5*worldSize[0]
		sincZ := sinc(bw[0] * z)
		for j := 0; j < meshSize[1]; j++ {
			y := float64(j)*cellSize[1] - 0.5*worldSize[1]
			sincY := sinc(bw[1] * y)
			for i := 0; i < meshSize[2]; i++ {
				x := float64(i)*cellSize[2] - 0.5*worldSize[2]
				sincX := sinc(bw[2] * x)
				bMask.Array[Z][k][j][i] = float32(sincX * sincY * sincZ)
			}
		}
	}

	a.SetMask("B_ext", bMask)
	Debug("Done.")

	//~ create FFT plan
	fftBuffer := new(gpu.Array)
	fftOutputSize := gpu.FFTOutputSize(logicSize)
	fftBuffer.Init(1, fftOutputSize, gpu.DO_ALLOC)
	plan := gpu.NewDefaultFFT(meshSize, logicSize)

	//~ get dimensions for per component quantities, lame
	OutputSize := make([]int, 3)
	OutputSize[0] = fftOutputSize[0]
	OutputSize[1] = fftOutputSize[2] / 2
	OutputSize[2] = fftOutputSize[1]

	//~ create two output quantities
	mFFTHostX := NewQuant("m-X", COMP, OutputSize, FIELD, Unit("A/m"), true, "FFT of M, real part")
	mFFTHostXArray := mFFTHostX.Buffer()

	mFFTHostY := NewQuant("m-Y", COMP, OutputSize, FIELD, Unit("A/m"), true, "FFT of M, imaginary part")
	mFFTHostYArray := mFFTHostY.Buffer()

	//~ Loop over the range of frequencies
	for i := 0; i < steps; i++ {
		//~ Get frequency of excitation
		f := fmin + float64(i)*df
		Debug(fmt.Sprintf("Calculating response for %g GHz...", f*1e-9))

		//~ Get time constant as 6T

		tp := 1.0 / f // period of RF

		tc := 6.0 * tp // length of excitation

		rt := 2.0 * tp // rise time of the excitation
		ft := 2.0 * tp // fall time of the excitation

		dt := tc / float64(FINE_T)
		//~ Generate CW excitation
		for ii := 0; ii < FINE_T; ii++ {
			t := float64(ii) * dt
			val := B_EXT * math.Sin(2.0*math.Pi*f*t) * (1.0 - math.Exp(-t/rt)) * (1.0 - math.Exp((t-tc)/ft))
			a.SetPointwise("B_ext", t, []float64{B0[0], B0[1], B0[2] + val})
		}
		a.SetPointwise("B_ext", float64(FINE_T)*dt, []float64{B0[0], B0[1], B0[2]})
		a.SetPointwise("B_ext", 9999.9, []float64{B0[0], B0[1], B0[2]})

		//~ run simulations
		a.Run(2.0 * tc)
		qM.Update()

		//~ substrate ground state
		gpu.Madd(m, m, m0, float32(-1.0))

		//~ Do FFT of m to buffer
		//~ split cmplx buffer to two per-component buffers
		//~ Do representation conversion if required
		Debug("Do FFT...")
		for ii := 0; ii < COMP; ii++ {
			plan.Forward(m.Component(ii), fftBuffer)
			extractCmplxComponents(fftBuffer.LocalCopy(), mFFTHostXArray.Component(ii), mFFTHostYArray.Component(ii), format)
		}
		Debug("Done.")

		//~ Save result to files
		filename := a.Engine.AutoFilename(mFFTHostX.Name(), OUT_FORMAT)
		a.Engine.SaveAs(mFFTHostX, OUT_FORMAT, []string{}, filename)
		filename = a.Engine.AutoFilename(mFFTHostY.Name(), OUT_FORMAT)
		a.Engine.SaveAs(mFFTHostY, OUT_FORMAT, []string{}, filename)

		//~ recover state
		a.SetS("t", 0.0)
		a.SetS("dt", 1e-15)
		a.SetV("B_ext", B0)
		m.CopyFromDevice(m0)
		qM.Invalidate()
	}
}

func sinc(arg float64) float64 {
	res := 1.0
	if arg != 0.0 {
		res = math.Sin(arg) / arg
	}
	return res
}

func extractCmplxComponents(src, comp1, comp2 *host.Array, format int) {
	//~ for i := range comp1.List {
	//~ R := float64(src.List[2*i+0])
	//~ I := float64(src.List[2*i+1])
	//~ switch format {
	//~ case 0:
	//~ tR := R
	//~ R = math.Sqrt(R*R + I*I)
	//~ I = math.Atan2(I, tR)
	//~ case 1:
	//~ // pass through
	//~ }
	//~ comp1.List[i] = float32(R)
	//~ comp2.List[i] = float32(I)
	//~ }
	for k := 0; k < comp1.Size4D[1]; k++ {
		for j := 0; j < comp1.Size4D[2]; j++ {
			for i := 0; i < comp1.Size4D[3]; i++ {
				R := float64(src.Array[0][k][i][2*j+0])
				I := float64(src.Array[0][k][i][2*j+1])
				switch format {
				case 0:
					tR := R
					R = math.Sqrt(R*R + I*I)
					I = math.Atan2(I, tR)
				case 1:
					// pass through
				}
				comp1.Array[0][k][j][i] = float32(R)
				comp2.Array[0][k][j][i] = float32(I)

			}
		}
	}
}
