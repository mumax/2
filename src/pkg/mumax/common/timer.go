//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2011  Arne Vansteenkiste and Ben Van de Wiele.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.


package common

import (
	"time"
	"fmt"
)

// Non-thread safe timer for debugging.
type Timer struct {
	StartNanos, TotalNanos int64 // StartTime == 0: not running
	Count                  int
}

func (t *Timer) StartTimer() {
	if t.StartNanos != 0 {
		panic(Bug("Timer.Start: already running"))
	}
	t.StartNanos = time.Nanoseconds()
}

func (t *Timer) StopTimer() {
	if t.StartNanos == 0 {
		panic(Bug("Timer.Stop: not running"))
	}
	t.TotalNanos += (time.Nanoseconds() - t.StartNanos)
	t.Count++
	t.StartNanos = 0
}

// Returns the total number of seconds this timer has been running.
// Correct even if the timer is running wh
func (t *Timer) Seconds() float64 {
	if t.StartNanos == 0 { //not running for the moment
		return float64(t.TotalNanos) / 1e9
	} // running for the moment
	return float64(t.TotalNanos+time.Nanoseconds()-t.StartNanos) / 1e9
}

func (t *Timer) Average() float64 {
	return t.Seconds() / (float64(t.Count))
}

func (t *Timer) TimerString() string {
	return fmt.Sprint(t.Seconds(), "s")
}
