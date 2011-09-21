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

type Timer struct {
	StartNanos, TotalNanos int64 // StartTime == 0: not running
	Count                  int
}

func (s *Timer) StartTimer() {
	if s.StartNanos != 0 {
		panic(Bug("Timer.Start: already running"))
	}
	s.StartNanos = time.Nanoseconds()
}

func (s *Timer) StopTimer() {
	if s.StartNanos == 0 {
		panic(Bug("Timer.Stop: not running"))
	}
	s.TotalNanos += (time.Nanoseconds() - s.StartNanos)
	s.Count++
	s.StartNanos = 0
}

func (t *Timer) Seconds() float64 {
	return float64(t.TotalNanos) / 1e9
}

func (t *Timer) Average() float64 {
	return t.Seconds() / (float64(t.Count))
}

func (t *Timer) String() string {
	return fmt.Sprintf("%4gms/invocation", t.Average()/1000)
}
