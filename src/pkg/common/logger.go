//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package common

// This file implements a global logger that prints to screen (stderr) and to a log file.
// The screen output can be filtered/disabled, while all output always goes to the log file.
// Author: Arne Vansteenkiste

import (
	"log"
	"os"
)

// INTERNAL global logger
var logger Logger

// INTERNAL 
type Logger struct {
	ShowDebug bool        // Include debug messages in stderr output?
	ShowWarn  bool        // Include warnings in stderr output?
	ShowPrint bool        // Include normal output in stderr output?
	Screen    *log.Logger // Logs to the screen (stderr), usually prints only limited output
	File      *log.Logger // Logs to a log file, usually prints all output (including debug)
}

// Initiates the logger and sets the log file.
// logfile == "" disables logging to file.
func InitLogger(logfile string) {
	logger.Init(logfile)
}

// INTERNAL Initiates the logger and sets a log file.
func (l *Logger) Init(logfile string) {
	l.Screen = log.New(os.Stderr, "", 0)
	if logfile != "" {
		out := FOpen(logfile)
		l.File = log.New(out, "", log.Ltime|log.Lmicroseconds)
		Debug("Opened log file:", logfile)
	}
}

// Log a debug message.
func Debug(msg ...interface{}) {
	if logger.ShowDebug {
		logger.Screen.Println(msg...)
	}
	LogFile(msg...)
}

const MSG_WARNING = "Warning:"

// Log a warning.
func Warning(msg ...interface{}) {
	if logger.ShowWarn {
		logger.Screen.Println(msg...)
	}
	LogFile(msg...)
}

// Log normal output.
func Println(msg ...interface{}) {
	if logger.ShowPrint {
		logger.Screen.Println(msg...)
	}
	LogFile(msg...)
}

// Log to the log file only.
func LogFile(msg ...interface{}) {
	if logger.File != nil {
		logger.File.Println(msg...)
	}
}
