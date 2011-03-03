//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package engine

// This file implements a 2-way pipe similar to io.Pipe(),
// but each end can read AND write.
// It can be used to connect an server and client engine.RPC.
// Author: Arne Vansteenkiste

import(
	"io"
)

type PipeReadWriter struct{
	Reader *io.PipeReader
	Writer *io.PipeWriter
}

func Pipe2() (end1, end2 PipeReadWriter){
	end1.Reader, end2.Writer = io.Pipe()
	end2.Reader, end1.Writer = io.Pipe()
}
