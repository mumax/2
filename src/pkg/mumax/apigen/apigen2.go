//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

// This package implements automated mumax API generation.
// Based on the exported methods of engine.API, an API
// library in any of the supported programming languages is
// automatically generated.
//
// Author: Arne Vansteenkiste
package apigen

import (
	. "mumax/common"
	"io/ioutil"
	"strings"
	"fmt"
	"os"
)

// Auto-generate API libraries for all languages.
func APIGen2() {
	// Read api.go
	buf, err := ioutil.ReadFile(GetExecDir() + "../src/pkg/mumax/engine/api.go")
	CheckIO(err)
	file := string(buf)

	// 
	lines := strings.Split(file, "\n")
	for i,line := range lines {
		if strings.HasPrefix(line, "func") {
			funcline := lines[i] // line that starts with func...
			comment := "#" // comments above func, with python doc comment ##
			j := i - 1
			for strings.HasPrefix(lines[j], "//") {
				comment += "#" + lines[j][2:] + "\n"
				j--
			}
			if j==i-1{ // no comment string
					comment = "##\n"
			}
			fmt.Println(comment + parseFunc(funcline), "\n")
		}
	}
}


func parseFunc(line string) (str string) {
	defer func(){
		err := recover()
		if err != nil{
			debug("not parsing", line)
			str = ""
		}
	}()

	//func (a API) Name (args) {

	name := line[index(line,')',1)+1:index(line,'(',2)]
	name = strings.Trim(name, " ")
	args := line[index(line,'(',2)+1:index(line,')',2)]
	args = parseArgs(args)
	str = name + "(" + args + "):"
	return 
}


func parseArgs(line string) string{
		parsed := ""
		args := strings.Split(line, ",")
		for i:=range args{
				args[i] = strings.Trim(args[i], " ")
				words := strings.Split(args[i], " ")
				if i !=0 {parsed += ", "}
				parsed +=  words[0]
		}
		return parsed
}

func debug(msg ...interface{}){
	fmt.Fprintln(os.Stderr, msg...)
}


// index of nth occurrence of sep in s.
func index(s string, sep uint8, n int) int{
	for i := range s{
			if s[i] == sep{n--}
			if n==0{return i}
	}
	return -1
}
