//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package refsh

import (
	"strings"
)

// INTERNAL
// Resolves a function name
func (r *Refsh) resolve(funcname string) Caller {
	funcname = strings.ToLower(funcname) // be case-insensitive
	for i := range r.Funcnames {
		if r.Funcnames[i] == funcname {
			return r.Funcs[i]
		}
	}
	return nil
}
