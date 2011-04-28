//  This file is part of MuMax, a high-perfomrance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package common

// Author: Arne Vansteenkiste

import (
	"testing"
)


func TestPipe2(t *testing.T) {
	end1, end2 := Pipe2()
	in := []byte{1, 2, 3, 4}
	out := []byte{0, 0, 0, 0}

	go end1.Write(in)
	end2.Read(out)

	for i := range out {
		if out[i] != byte(i)+1 {
			t.Fail()
		}
	}

	end1.Close()
	end2.Close()
}
