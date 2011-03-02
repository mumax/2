//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package refsh

import (
	"testing"
	"os"
	"fmt"
)

func TestTokenizer(test *testing.T) {
	in, err := os.Open("test.in", os.O_RDONLY, 0666)
	if err != nil {
		test.Fail()
		return
	}

	for line, eof := ReadNonemptyLine(in); !eof; line, eof = ReadNonemptyLine(in) {
		fmt.Println(line)
	}
}
