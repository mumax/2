//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any
//  copyright notices and prominently state that you modified it, giving a relevant date.

package main

import (
	"fmt"
	"log"
	"mumax/frontend"
	_ "mumax/modules" // register and link core modules
	_ "mumax/ovf"     // register and link OOMMF output formats
	"os"
	"path/filepath"
	"runtime"
)

const (
	PYLIB            = "PYTHONPATH"
	PYTHONMODULEPATH = "/../src/python"
)

func main() {

	// setup enviroment
	mumaxBinDir, err := filepath.Abs(filepath.Dir(os.Args[0]))
	if err != nil {
		log.Fatal(err)
	}

	envValueSep := ":"
	if runtime.GOOS == "windows" {
		envValueSep = ";"
	}

	pyLibValue := os.Getenv(PYLIB)
	pyLibValue += (envValueSep + mumaxBinDir + PYTHONMODULEPATH)
	pyLibValue = filepath.FromSlash(pyLibValue)
	os.Setenv(PYLIB, pyLibValue)

	frontend.Main()
}
