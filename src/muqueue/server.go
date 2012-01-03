//  This file is part of muQueue, a job scheduler for MuMax.
//  Copyright 2012  Arne Vansteenkiste.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.

package main

// Server main loop

import ()

func serverMain() {
	initSched()
	go runSched()
	initNet()
	runNet() // loops forever
}
