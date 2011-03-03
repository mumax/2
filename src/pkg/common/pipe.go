//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package common

// This file implements a 2-way pipe similar to io.Pipe(),
// but each end can read AND write. I.e. each end implements
// io.ReadWriteCloser.
// This can be used, e.g.,  to connect an server and client engine.RPC.
//
// Author: Arne Vansteenkiste

import (
	"io"
	"os"
)

// Returns a 2-way pipe. Each end implements io.ReadWriteCloser.
func Pipe2() (end1, end2 *PipeReadWriter) {
	end1 = new(PipeReadWriter)
	end2 = new(PipeReadWriter)
	end1.Reader, end2.Writer = io.Pipe()
	end2.Reader, end1.Writer = io.Pipe()
	return
}

// INTERNAL
// 2-way pipe
type PipeReadWriter struct {
	Reader io.ReadCloser
	Writer io.WriteCloser
}

// Implements io.Reader
func(p *PipeReadWriter) Read(b []byte) (n int, err os.Error){
	return p.Reader.Read(b)
}

// Implements io.Writer
func(p *PipeReadWriter) Write(b []byte) (n int, err os.Error){
	return p.Writer.Write(b)
}


// Implements io.Closer
func(p *PipeReadWriter) Close() (err os.Error){
	err = p.Reader.Close()
	err = ErrCat(err, p.Writer.Close()) // Combine the two errors into one.
	return
}
