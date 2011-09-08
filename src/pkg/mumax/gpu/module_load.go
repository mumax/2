//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package gpu

// This file implements loading .ptx modules.
// Author: Arne Vansteenkiste

import (
	. "mumax/common"
	cu "cuda/driver"
	"strings"
	"fmt"
	"path"
)

// INTERNAL
var (
	_modules   map[string]cu.Module   // maps a module name (file name without .ptx) on the module
	_functions map[string]cu.Function // maps a function name on the function
	_funcArgs  map[string][]argInfo   // maps a function name on a list with the argument types and names.
)

func init() {
	_modules = make(map[string]cu.Module)
	_functions = make(map[string]cu.Function)
	_funcArgs = make(map[string][]argInfo)
}

// Loads a .ptx module for all GPUs.
func LoadModule(filename string) {
	Debug("Loading module: ", filename)
	Assert(strings.HasSuffix(filename, ".ptx"))

	// load the module into _modules
	fname := path.Base(filename)
	name := fname[:len(fname)-len(".ptx")] // module name without .ptx
	_, ok := _modules[name]
	if ok {
		panic(Bug(fmt.Sprintf(ERR_MODULE_LOADED, fname)))
	}
	module := cu.ModuleLoad(filename)
	_modules[name] = module

	// load all functions into _functions
	funcArgs := parsePTXArgTypes(filename)
	for funcName := range funcArgs {
		_, ok := _functions[funcName]
		if ok {
			panic(Bug(fmt.Sprintf(ERR_FUNC_REDEFINED, funcName, fname)))
		}
		_functions[funcName] = module.GetFunction(funcName)
		_funcArgs[funcName] = funcArgs[funcName]
	}
}

// Loads the .ptx module only when it has not yet been loaded before.
func assureModule(modname string) {
	_, ok := _modules[modname]
	if !ok {
		LoadModule(findModule(modname))
	}
}

// Error message
const (
	ERR_MODULE_LOADED  = "module already loaded: %s"
	ERR_FUNC_REDEFINED = "function already defined: %s (%s)"
)
