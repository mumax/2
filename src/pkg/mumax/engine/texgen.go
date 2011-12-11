//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package engine

// This file automatically generates LaTeX documentation
// for all registered modules.

// Author: Arne Vansteenkiste

import (
	. "mumax/common"
	"strings"
)

func TexGen(){

	for mod := range modules{
		moduleTexGen(mod)
	}
}


func moduleTexGen(module string){
	out := OpenWRONLY("modules/" + texify(module) + ".tex")
	defer out.Close()

	
}

func texify(str string)string{
	const ALL=-1
	str =strings.Replace(str, "/", "-", ALL)
	return str
}
