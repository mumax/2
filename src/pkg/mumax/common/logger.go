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
	"fmt"
)

// INTERNAL global logger
var logger Logger

// INTERNAL 
type Logger struct {
	ShowDebug   bool        // Include debug messages in stderr output?
	ShowWarn    bool        // Include warnings in stderr output?
	ShowPrint   bool        // Include normal output in stderr output?
	Screen      *log.Logger // Logs to the screen (stderr), usually prints only limited output
	File        *log.Logger // Logs to a log file, usually prints all output (including debug)
	Initialized bool        // If the logger is not initialized, dump output to stderr.
}

// Initiates the logger and sets the log file.
// logfile == "" disables logging to file.
func InitLogger(logfile string, options ...LogOption) {
	logger.Init(logfile, options...)
}

type LogOption int

const (
	LOG_NOSTDOUT LogOption = 1 << iota
	LOG_NODEBUG  LogOption = 1 << iota
	LOG_NOWARN   LogOption = 1 << iota
)

// INTERNAL Initiates the logger and sets a log file.
func (l *Logger) Init(logfile string, options ...LogOption) {
	opt := 0
	for i := range options {
		opt |= int(options[i])
	}
	l.Screen = log.New(os.Stderr, "", 0)

	l.ShowDebug = opt&int(LOG_NODEBUG) == 0
	l.ShowWarn = opt&int(LOG_NOWARN) == 0
	l.ShowPrint = opt&int(LOG_NOSTDOUT) == 0
	if logfile != "" {
		out := FOpen(logfile)
		l.File = log.New(out, "", log.Ltime|log.Lmicroseconds)
		//Debug("Opened log file:", logfile)
	}
	l.Initialized = true
	//Log("log normal output:", l.ShowPrint)
	//Log("log debug messages:", l.ShowDebug)
	//Log("log warnings:", l.ShowWarn)
}

// Log a debug message.
func Debug(msg ...interface{}) {
	if !logger.Initialized && logger.ShowDebug {
		fmt.Fprintln(os.Stderr, msg...)
	}
	if logger.ShowDebug {
		logger.Screen.Println(msg...)
	}
	LogFile(msg...)
}

const MSG_WARNING = "Warning:"

// Log a warning.
func Warn(msg ...interface{}) {
	if !logger.Initialized && logger.ShowWarn {
		fmt.Fprintln(os.Stderr, msg...)
	}
	if logger.ShowWarn {
		logger.Screen.Println(msg...)
	}
	LogFile(msg...)
}

// Log normal output.
func Log(msg ...interface{}) {
	if !logger.Initialized {
		fmt.Fprintln(os.Stderr, msg...)
	}
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
