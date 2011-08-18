//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package frontend

// The client implements Inter-Process-Communication
// between mumax and a scripting language.


import (
	. "mumax/common"
	"mumax/engine"
	"fmt"
	"io"
	"exec"
	"os"
	"time"
)


type Client struct {
	inputFile, outputDir string
	ipc                  jsonRPC
	eng                  *engine.Engine
	api                  engine.API
	infifo, outfifo      *os.File
	cleanfiles           []string // list of files to be deleted upon program exit
}


// Initializes the mumax client to parse infile, write output
// to outdir and connect to a server over conn.
func (c *Client) Init(inputFile, outputDir, command string) {
	c.outputDir = outputDir
	c.inputFile = inputFile
	c.eng = engine.NewEngine()
	c.eng.InitMicromagnetism()
	Debug(c.eng.String())
	// dot...
	c.api = engine.API{c.eng}
	//c.ipc.Init(c.api)
}


func (c *Client) Run() {
	c.makeFifos() // make the FIFOs but do not yet try to open them
	command, waiter := c.startSubcommand()
	ok := c.handshake(waiter)
	if !ok {
		panic(InputErr(fmt.Sprint("subcommand ", command, " exited prematurely")))
	}
	c.openFifos()
	//c.interpretCommands()
	c.ipc.Init(c.infifo, c.outfifo, c.api)
	c.ipc.Run()

	// wait for the sub-command to exit
	Debug("Waiting for subcommand ", command, "to exit")
	exitstat := <-waiter

	if exitstat != 0 {
		panic(InputErr(fmt.Sprint(command, " exited with status ", exitstat)))
	}
}


//type VoidArgs struct{}

// run the sub-command (e.g. python) to interpret the script file
// it will first hang while trying to open the FIFOs
func (c *Client) startSubcommand() (command string, waiter chan (int)) {

	//CheckErr(os.Setenv("PYTHONPATH", os.Getenv("PYTHONPATH")+":"+path.Clean(GetExecDir())), ERR_IO)
	//CheckErr(os.Setenv("CLASSPATH", os.Getenv("CLASSPATH")+":"+path.Clean(GetExecDir())), ERR_IO)
	CheckErr(os.Setenv("MUMAX2_OUTPUTDIR", c.outputDir), ERR_IO)

	var args []string
	command, args = commandForFile(c.inputFile) // e.g.: "python"

	proc := exec.Command(command, args...) //:= subprocess(command, args)

	stderr, err4 := proc.StderrPipe()
	CheckErr(err4, ERR_IO)
	stdout, err5 := proc.StdoutPipe()
	CheckErr(err5, ERR_IO)
	CheckErr(proc.Start(), ERR_IO)
	go logStream("["+command+"]", stderr, true)
	go logStream("["+command+"]", stdout, false)

	Debug(command, "PID:", proc.Process.Pid)
	// start waiting for sub-command asynchronously and
	// use a channel to signal sub-command completion
	waiter = make(chan (int))
	go func() {
		exitstat := 666 // dummy value 
		err := proc.Wait()
		if err != nil {
			if msg, ok := err.(*os.Waitmsg); ok {
				exitstat = msg.ExitStatus()
			} else {
				panic(InputErr(err.String()))
			}
		} else {
			exitstat = 0
		}
		waiter <- exitstat // send exit status to signal completion 
	}()
	// pipe sub-command output to the logger
	return
}


// open FIFOs for communication
// there is a synchronization subtlety here:
// opening the fifo's blocks until they have been
// opened on the other side as well. So the subprocess
// must be started first and must open the fifos in
// the correct order (first OUT then IN).
// this function hangs when the subprocess does not open the fifos.
func (c *Client) openFifos() {
	Debug("Opening FIFOs will block until child process opens the other end")
	var err os.Error
	c.outfifo, err = os.OpenFile(c.outputDir+"/"+OUTFIFO, os.O_WRONLY, 0666)
	CheckErr(err, ERR_BUG)
	c.infifo, err = os.OpenFile(c.outputDir+"/"+INFIFO, os.O_RDONLY, 0666)
	CheckErr(err, ERR_IO)
	return
}


// read text commands from infifo, execute them and return the result to outfifo
// stop when a fifo gets closed by the other end
//func (c *Client) interpretCommands() {
//
//	// interpreter executes commands from subprocess
//	for line, eof := parseLine(c.infifo); !eof; line, eof = parseLine(c.infifo) {
//		// call locally
//		ret := c.ipc.Call(line[0], line[1:])
//		// pass return value to subprocess
//		switch len(ret) {
//		default:
//			panic(Bug("Method returned too many values"))
//		case 0:
//			fmt.Fprintln(c.outfifo)
//		case 1:
//			fmt.Fprintln(c.outfifo, ret[0])
//		}
//	}
//}


// wait until the subcommand creates the handshake file,
// indicating that it will open the fifos. however, if the
// the subprocess exits before creating the handshake file,
// return not OK. in that case we should not attempt to ope
// the fifos because they will block forever.
func (c *Client) handshake(procwaiter chan (int)) (ok bool) {
	Debug("waiting for handshake")
	filewaiter := pollFile(c.outputDir + "/" + HANDSHAKEFILE)
	select {
	case <-filewaiter:
		return true
	case exitstat := <-procwaiter:
		Debug("Child command exited with status ", exitstat)
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


// pipes standard output/err of the command to the logger
// typically called in a separate goroutine
func logStream(prefix string, in io.Reader, error bool) {
	var bytes [BUFSIZE]byte
	buf := bytes[:]
	var err os.Error = nil
	n := 0
	for err == nil {
		n, err = in.Read(buf)
		if n != 0 {
			if error {
				Err(prefix, string(buf[:n]))
			} else {
				Log(prefix, string(buf[:n]))
			}
		} // TODO: no printLN
	}
}

// IO buffer size
const BUFSIZE = 4096


// makes the FIFOs for inter-process communications
func (c *Client) makeFifos() {
	outfname := c.outputDir + "/" + OUTFIFO
	infname := c.outputDir + "/" + INFIFO
	c.cleanfiles = append(c.cleanfiles, infname, outfname)
	mkFifo(infname)
	mkFifo(outfname)
}
