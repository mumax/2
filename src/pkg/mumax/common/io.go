//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package common

// This file implements convenience functions for opening files.
// Errors are wrapped in mumax IOErr's and cause a panic.

import (
	"os"
	"path"
)


// Opens a file for read-write.
// Truncates existing file or creates the file if neccesary.
// The permission is the same as the parent directory
// (but without executability).
func FOpen(filename string) *os.File {
	perm := Permission(Parent(filename))
	perm &= MASK_NO_EXEC
	file, err := os.Create(filename)
	if err != nil {
		panic(IOErr(err.String()))
	}
	return file
}

// Permission flag for rw-rw-rw
const MASK_NO_EXEC = 0666


// returns the parent directory of a file
func Parent(filename string) string {
	dir, _ := path.Split(filename)
	if dir == "" {
		dir = "."
	}
	return dir
}


// returns the file's permissions
func Permission(filename string) uint32 {
	stat, err := os.Stat(filename)
	if err != nil {
		panic(IOErr(err.String()))
	}
	return stat.Permission()
}
