//  This file is part of muQueue, a job scheduler for MuMax.
//  Copyright 2012  Arne Vansteenkiste.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.

package main

// User

import ()

// User
type User struct {
	name  string
	group string
}

var users map[string]*User = make(map[string]*User)

func GetUser(name string) *User {
	if user, ok := users[name]; ok {
		return user
	}
	panic("No such user: " + name)
	return nil // silence 6g
}

func (u *User) String() string {
	return u.name
}
