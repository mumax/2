//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package common

// This file implements parsing of PTX assembly files to extract the function argument types.
// Author: Arne Vansteenkiste

import (
	"io/ioutil"
	"strings"
	//"fmt"
)


// Parses the PTX file and returns a map with the argument types for each function.
// E.g.:
// 	void myFunc(int a, float b, void* ptr)
// gives:
// 	{"myFunc":[]int{1, 2, 6}}
// Where 1 represents int, 2 float, 6 pointer.
func parsePTXArgTypes(fname string) map[string][]int {
	defer func() {
		err := recover()
		if err != nil {
			panic(Bug("Error parsing " + fname))
		}
	}()

	types := make(map[string][]int)
	content, err := ioutil.ReadFile(fname)
	CheckErr(err, ERR_IO)
	words := strings.Split(string(content), " ", -1)
	for i, word := range words {
		//fmt.Println(i, word)
		if strings.HasSuffix(word, ".param") {
			typ := words[i+1][1:len(words[i+1])]     // e.g. ".s32"
			name := words[i+2]                       // e.g. "__cudaparm_funcName_ArgName"
			funcArgName := name[len("__cudaparm_"):] // e.g. "funcName_ArgName"
			cut := strings.Index(funcArgName, "_")
			funcname := funcArgName[:cut]
			typeId, ok := ptxTypeId[typ]
			if !ok {
				panic(Bug("PTX type " + typ))
			}
			types[funcname] = append(types[funcname], typeId)
		}
	}

	return types
}


// Enumerates PTX argument types.
const (
	invalid = iota
	s32
	s64
	f32
	f64
	u32
	u64
)

// String to PTX argument type number.
var ptxTypeId map[string]int = map[string]int{"s32": s32, "s64": s64, "f32": f32, "f64": f64, "u32": u32, "u64": u64}
