//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package common

// This file implements miscellaneous common functions.
// Author: Arne Vansteenkiste

import (
	"path"
	"os"
)


// Go equivalent of &array[index] (for a float array).
func ArrayOffset(array uintptr, index int) uintptr {
	return uintptr(array + uintptr(SIZEOF_CFLOAT*index))
}

// Size, in bytes, of a C single-precision float
const SIZEOF_CFLOAT = 4


// Replaces the extension of filename by a new one.
func ReplaceExt(filename, newext string) string {
	extension := path.Ext(filename)
	return filename[:len(filename)-len(extension)] + newext
}


// Gets the directory where the executable is located.
func GetExecDir() string {
	dir, err := os.Readlink("/proc/self/exe")
	CheckErr(err, ERR_IO)
	return Parent(dir)
}


// Combines two Errors into one.
// If a and b are nil, the returned error is nil.
// If either is not nil, it is returned.
// If both are not nil, the first one is returned.
func ErrCat(a, b os.Error) os.Error {
	if a != nil {
		return a
	}
	if b != nil {
		return b
	}
	return nil
}
