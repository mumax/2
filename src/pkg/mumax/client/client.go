//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package client

import ()

type Client struct {
}


// For testing purposes.
func (c *Client) Version() int {
	return 2
}

// For testing purposes.
func (c *Client) Echo(i int) int {
	return i
}

// For testing purposes.
func (c *Client) Sum(i, j int) int {
	return i + j
}
