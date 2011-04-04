//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package common

import (
	"path"
)


// Replaces the extension of filename by a new one.
func ReplaceExt(filename, newext string) string {
	extension := path.Ext(filename)
	return filename[:len(filename)-len(extension)] + newext
}

