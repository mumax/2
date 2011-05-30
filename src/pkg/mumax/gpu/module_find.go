//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package gpu

// This file implements looking for .ptx module files
// Author: Arne Vansteenkiste

import (
	. "mumax/common"
)

func init() {
	_lookPath = GetExecDir() 
}


// INTERNAL: where to look for .ptx files.
var _lookPath string


// Finds the full path of a module specified by name, using _lookPath.
func findModule(modname string) (filename string) {
	fname := modname + ".ptx"
	filename = _lookPath + fname
	return

}
