//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package frontend

// This file implements the mumax2's main function.
// Arne Vansteenkiste

import (
	cu "cuda/driver"
	"fmt"
	. "mumax/common"
	"os"
	"os/exec"
	"path"
	"runtime"
	"net"
	"bufio"
	"strings"
)

// run the input files given on the command line
func clientMain() {
	//defer fmt.Println(RESET)

	if !*flag_silent {
		fmt.Println(WELCOME)
	}

	//infile := inputFile()
	//outdir := outputDir(infile)
	outdir := "."
	
	initOutputDir(outdir)
	initLogger(outdir)
	LogFile(WELCOME)
	hostname, _ := os.Hostname()
	Debug("Hostname:", hostname)
	Debug("Go", runtime.Version())
	//command := *flag_command

	// initialize CUDA first
	Debug("Initializing CUDA")
	runtime.LockOSThread()
	Debug("Locked OS Thread")
	
	cu.Init(0)
	
	initMultiGPU()
	
	if *flag_test {
		testMain()
		return
	}
	
	m_addr := SERVERADDR + ":" + PORT	
	Debug("Starting local MuMax server on " + m_addr + " ...")
	
	ln, err := net.Listen("tcp", m_addr)
	if err != nil {
		panic("Cannot start master MuMax server!") 
	}
	Debug("Done.")
	
	for {
		Debug("Waiting for clients...")
		wire, err1 := ln.Accept()
		if err1 != nil {
			Debug("[WARNING] One of the clients has failed to connect.")
			continue
		}	
		go ServeClient(wire)
	}	
}

// return the output directory
func outputDir(inputFile string) string {
	if *flag_outputdir != "" {
		return *flag_outputdir
	}
	return inputFile + ".out"
}

// make the output dir
func initOutputDir(outputDir string) {
	if *flag_force {
		err := exec.Command("rm", "-rf", outputDir).Run() //syscommand("rm", []string{"-rf", outputDir})
		if err != nil {
			Log("rm -rf", outputDir, ":", err)
		}
	}
	if outputDir !="." {
		os.Mkdir(outputDir, 0777)
	}
	
	/*errOut := os.Mkdir(outputDir, 0777)
	if outputDir != "." {
		CheckIO(errOut)
	} else {
		Log(errOut)
	}*/
	
}

// Gets the response from the client and starts the slave server
func ServeClient(wire net.Conn) {

	m_in := bufio.NewReader(wire)
	m_out := bufio.NewWriter(wire)
	
	// Reads Client's name and path to the script file if any
	
	cmsg, err := m_in.ReadString('\n')
	if err != nil {
		Debug("[WARNING] Client has failed to send its name and path to the script")
		return
	}
	
	cmsg_slice := strings.Split(cmsg,":")
	
	ClientName := cmsg_slice[0]	
	// This approach is not universal at all
	ClientPath := strings.TrimSpace(cmsg_slice[1]) + ".out"
		
	Debug(ClientName + " is connected")	
	Debug("Client asks to write into: " + ClientPath)
	
	s_ln, err := net.Listen("tcp", "localhost:0")
	if err != nil {
		Debug("[WARNING] MuMax has failed to start slave server")
		return
	}
	
	s_addr	:= s_ln.Addr().String()
	Debug("The slave server is started on: " + s_addr)
	addr_msg := s_ln.Addr().String() + EOM
	m_out.WriteString(addr_msg)
	m_out.Flush()
	
	Debug("Waiting " + ClientName + " to respond to slave server...")	
	s_wire, err := s_ln.Accept()
	if err != nil {
		Debug("[WARNING]" + ClientName + " has failed to connect")
		return
	}	
	Debug("Done.")
	
	var client Client	
	client.wire = s_wire
	client.Init(ClientPath)
	client.Run()
}

// initialize the logger
func initLogger(outputDir string) {
	var opts LogOption
	if !*flag_debug {
		opts |= LOG_NODEBUG
	}
	if *flag_silent {
		opts |= LOG_NOSTDOUT | LOG_NODEBUG | LOG_NOWARN
	}
	if !*flag_warn {
		opts |= LOG_NOWARN
	}

	logFile := *flag_logfile
	if logFile == "" {
		logFile = outputDir + "/mumax2.log"
	}
	InitLogger(logFile, opts)
	Debug("Logging to", logFile)
}

// given a file name (e.g. file.py)
// this returns a command to run the file (e.g. python file.py, java File)
func commandForFile(file string) (command string, args []string) {
	if *flag_command != "" {
		return *flag_command, []string{file}
	}
	if file == "" {
		panic(IOErr("no input file"))
	}
	switch path.Ext(file) {
	default:
		panic(InputErr("Cannot handle files with extension " + path.Ext(file)))
	case ".py":
		return "python", []string{file}
		//case ".java":
		//	return GetExecDir() + "javaint", []string{file}
		//case ".class":
		//	return "java", []string{ReplaceExt(file, "")}
		//case ".lua":
		//	return "lua", []string{file}
	}
	panic(Bug("unreachable"))
	return "", nil
}

const (
	INFIFO        = "in.fifo"   // FIFO filename for mumax->subprocess text-based function calls.
	OUTFIFO       = "out.fifo"  // FIFO filename for mumax<-subprocess text-based function calls.
	HANDSHAKEFILE = "handshake" // Presence of this file indicates subprocess initialization OK.
	PORT		  = "3655"
	SERVERADDR    = "localhost"
	EOM		      = "<<< End of mumax message >>>"
)
