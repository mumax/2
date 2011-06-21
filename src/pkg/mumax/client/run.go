//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package client

// This file implements Inter-Process-Communication
// between mumax and a scripting language.

import (
	. "mumax/common"
	"mumax/engine"
	"fmt"
	"net"
	"rpc"
	"path"
	"io"
	"os"
	"runtime"
	"time"
)


// run the input files given on the command line
// todo: split in smaller functions
func run() {
	initOutputDir()

	initLogger()
	Log(WELCOME)
	Debug("Go", runtime.Version())

	server := rpc.NewClient(engineConn())

	makeFifos(outputDir()) // make the FIFOs but do not yet try to open them

	command, waiter := startSubcommand()

	ok := handshake(waiter)
	if !ok {
		panic(InputErr(fmt.Sprint("subcommand ", command, " exited prematurely")))
	}

	infifo, outfifo := openFifos()

	interpretCommands(infifo, outfifo, server)

	// wait for the sub-command to exit
	Debug("Waiting for subcommand ", command, "to exit")
	exitstat := <-waiter

	if exitstat != 0 {
		panic(InputErr(fmt.Sprint(command, " exited with status ", exitstat)))
	}
}



func engineConn() io.ReadWriteCloser {
	if *Flag_engineAddr == "" { // no remote engine specified, use local one
			Debug("Connecting to local engine")
			return  engine.LocalConn()
	} else {
			Debug("Connecting to remote engine: ", *engine.Flag_net, *Flag_engineAddr)
		conn, err := net.Dial(*engine.Flag_net, *Flag_engineAddr)
		CheckErr(err, ERR_IO)
		return conn
	}
	panic(Bug("unreachable"))
	return nil
}


// make the output dir
func initOutputDir() {
	if *Flag_force {
		err := syscommand("rm", []string{"-rf", outputDir()}) // ignore errors.
		if err != nil {
			Log("rm -rf", outputDir(), ":", err)
		}
	}
	errOut := os.Mkdir(outputDir(), 0777)
	CheckErr(errOut, ERR_IO)

	// set the output dir in the environment so the child process can fetch it.
	CheckErr(os.Setenv("MUMAX2_OUTPUTDIR", outputDir()), ERR_IO)
}


// initialize the logger
func initLogger() {
	var opts LogOption
	if !*Flag_debug {
		opts |= LOG_NODEBUG
	}
	if *Flag_silent {
		opts |= LOG_NOSTDOUT | LOG_NODEBUG | LOG_NOWARN
	}
	if !*Flag_warn {
		opts |= LOG_NOWARN
	}

	logFile := *Flag_logfile
	if logFile == "" {
		logFile = outputDir() + "/mumax2.log"
	}
	InitLogger(logFile, opts)
}


// run the sub-command (e.g. python) to interpret the script file
// it will first hang while trying to open the FIFOs
func startSubcommand() (command string, waiter chan (int)) {

	os.Setenv("PYTHONPATH", os.Getenv("PYTHONPATH")+":"+path.Clean(GetExecDir()))
	os.Setenv("CLASSPATH", os.Getenv("CLASSPATH")+":"+path.Clean(GetExecDir()))

	var args []string
	command, args = commandForFile(inputFile()) // e.g.: "python"
	proc := subprocess(command, args)
	Debug(command, "PID:", proc.Process.Pid)
	// start waiting for sub-command asynchronously and
	// use a channel to signal sub-command completion
	waiter = make(chan (int))
	go func() {
		msg, err := proc.Wait(0)
		if err != nil {
			panic(InputErr(err.String()))
		}
		waiter <- msg.ExitStatus() // send exit status to signal completion 
	}()
	// pipe sub-command output to the logger
	go logStream("["+command+":err]", proc.Stderr)
	go logStream("["+command+":out]", proc.Stdout)
	return
}


// open FIFOs for communication
// there is a synchronization subtlety here:
// opening the fifo's blocks until they have been
// opened on the other side as well. So the subprocess
// must be started first and must open the fifos in
// the correct order (first OUT then IN).
// this function hangs when the subprocess does not open the fifos.
func openFifos() (infifo, outfifo *os.File) {
	Debug("Opening FIFOs will block until child process opens the other end")
	var err os.Error
	outfifo, err = os.OpenFile(outputDir()+"/"+OUTFIFO, os.O_WRONLY, 0666)
	CheckErr(err, ERR_BUG)
	infifo, err = os.OpenFile(outputDir()+"/"+INFIFO, os.O_RDONLY, 0666)
	CheckErr(err, ERR_IO)
	return
}


// read text commands from infifo, execute them and return the result to outfifo
// stop when a fifo gets closed by the other end
func interpretCommands(infifo, outfifo *os.File, server *rpc.Client) {
	// interpreter exports client methods
	c := new(Client)
	var ipc interpreter
	ipc.init(c)

	// interpreter executes commands from subprocess
	for line, eof := parseLine(infifo); !eof; line, eof = parseLine(infifo) {
		//Debug("call:", line)
		ret := ipc.call(line[0], line[1:])
		//Debug("return:", ret)
		switch len(ret) {
		default:
			panic(Bug("Method returned too many values"))
		case 0:
			fmt.Fprintln(outfifo)
		case 1:
			fmt.Fprintln(outfifo, ret[0])
		}
	}
}


// wait until the subcommand creates the handshake file,
// indicating that it will open the fifos. however, if the
// the subprocess exits before creating the handshake file,
// return not OK. in that case we should not attempt to ope
// the fifos because they will block forever.
func handshake(procwaiter chan (int)) (ok bool) {
	Debug("waiting for handshake")
	filewaiter := pollFile(outputDir() + "/" + HANDSHAKEFILE)
	select {
	case <-filewaiter:
		return true
	case exitstat := <-procwaiter:
		Log("Child command exited with status ", exitstat)
		return false
	}
	panic(Bug("unreachable"))
	return false
}


// returns a channel that will signal when the file has appeared
func pollFile(fname string) (waiter chan (int)) {
	waiter = make(chan (int))
	go func() {
		for !FileExists(fname) {
			time.Sleep(10)
		}
		waiter <- 1
	}()
	return
}

// given a file name (e.g. file.py)
// this returns a command to run the file (e.g. python file.py, java File)
func commandForFile(file string) (command string, args []string) {
	if *Flag_scriptcmd != "" {
		return *Flag_scriptcmd, []string{file}
	}
	switch path.Ext(file) {
	default:
		panic(InputErr("Cannot handle files with extension " + path.Ext(file)))
	case ".py":
		return "python", []string{file}
	case ".java":
		return "javaint", []string{file}
	case ".class":
		return "java", []string{ReplaceExt(file, "")}
	case ".lua":
		return "lua", []string{file}
	}
	panic(Bug("unreachable"))
	return "", nil
}


// pipes standard output/err of the command to the logger
// typically called in a separate goroutine
func logStream(prefix string, in io.Reader) {
	var bytes [BUFSIZE]byte
	buf := bytes[:]
	var err os.Error = nil
	n := 0
	for err == nil {
		n, err = in.Read(buf)
		if n != 0 {
			Log(prefix, string(buf[:n]))
		} // TODO: no printLN
	}
}

// IO buffer size
const BUFSIZE = 4096


// makes the FIFOs for inter-process communications
func makeFifos(outputDir string) {
	outfname := outputDir + "/" + OUTFIFO
	infname := outputDir + "/" + INFIFO
	cleanfiles = append(cleanfiles, infname, outfname)
	mkFifo(infname)
	mkFifo(outfname)

}


// Default FIFO filename for inter-process communication.
const (
	INFIFO        = "in.fifo"
	OUTFIFO       = "out.fifo"
	HANDSHAKEFILE = "handshake"
)
