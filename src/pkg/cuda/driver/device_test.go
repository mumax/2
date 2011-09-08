// Copyright 2011 Arne Vansteenkiste (barnex@gmail.com).  All rights reserved.
// Use of this source code is governed by a freeBSD
// license that can be found in the LICENSE.txt file.

package driver

import (
	"testing"
	"fmt"
)

func TestDevice(t *testing.T) {
	fmt.Println("CUDA device", CtxGetDevice())
	fmt.Println(CtxGetDevice().GetName())
}
