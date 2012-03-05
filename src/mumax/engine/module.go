//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package engine

import (
	. "mumax/common"
	"fmt"
)

// A physics module. Loading it adds various quantity nodes to the engine.
type Module struct {
	Name        string          // Name to identify to module to the machine
	Description string          // Human-readable description of what the module does
	LoadFunc    func(e *Engine) // Loads this module's quantities and dependencies into the engine
}

// Map with registered modules
var modules map[string]Module = make(map[string]Module)

// Registers a module in the list of known modules.
// Each module should register itself in its init() function.
func RegisterModule(name, description string, loadfunc func(e *Engine)) {
	if _, ok := modules[name]; ok {
		panic(InputErr("module " + name + " already registered"))
	}
	modules[name] = Module{name, description, loadfunc}
}

func GetModule(name string) Module {
	module, ok := modules[name]

	if !ok {
		opts := []string{}
		for k, _ := range modules {
			opts = append(opts, k)
		}
		panic(InputErr(fmt.Sprint("Unknown module:", name, " Options: ", opts)))
	}
	return module
}
