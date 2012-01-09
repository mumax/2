//  This file is part of muQueue, a job scheduler for MuMax.
//  Copyright 2012  Arne Vansteenkiste.
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.

package main

// User

import (
	"fmt"
)

// User
type User struct {
	name  string // username
	group string // user's group
	share int    // user's share (number of GPUs)
}

var users map[string]*User = make(map[string]*User)

func AddUser(name, group string, share int) {
	users[name] = &User{name, group, share}
}

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

func (u *User) LongString() string {
	return fmt.Sprint(u.name, " group:", u.group, " share:", u.share, "GPUs")
}
