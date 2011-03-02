//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package common

import(
	"log"
	"os"
)

// INTERNAL global logger
var logger Logger

type Logger struct{
	ShowDebug bool
	ShowWarn bool
	ShowPrint bool
	Screen *log.Logger // Logs to the screen (stderr), usually prints only limited output
	File *log.Logger // Logs to a log file, usually prints all output (including debug)
}

func(l *Logger) Init(logfile string){
	l.Screen = log.New(os.Stderr, "", log.Ltime | log.Lmicroseconds)
	//out = 
	//l.File = log.New(out, "", log.Ltime | log.Lmicroseconds)
}

func Debug(msg ...interface{}){
	if logger.ShowDebug{
		logger.Screen.Println(msg...)
	}
	if logger.File != nil{logger.File.Println(msg...)}
}

func Warn(msg ...interface{}){

}

func Println(msg ...interface{}){

}
