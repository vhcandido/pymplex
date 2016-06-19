#!/usr/bin/env python2

import numpy as np
import sys
import json

class Model:
    def __init__(self, data, debug=False):
        self.name = data['name']
        self.obj = data['obj']

        self.c = np.array(data['C'])

        self.R = data['inq']
        self.A = np.matrix(data['A'])
        self.b = np.array(data['b'])
        self.m, self.n = np.shape(self.A)

        self.x = np.array([])

        self.debug = debug
        self.B_i = {}

    def __standart_form(self):
        # change function objective to minimization
        if self.obj == 'max':
            self.obj = 'min'
            self.c *= -1

        # make b array non-negative
        for i in range(self.m):
            if self.b[i] < 0:
                self.A[i] *= -1
                self.b[i] *= -1
                if self.R[i] == '<=':
                    self.R[i] = '>='
                elif self.R[i] == '>=':
                    self.R[i] = '<='

        # add slack variables
        # count occurencies of inequalities
        slack = self.R.count('>=') + self.R.count('<=')

        # go back if no slack variables are needed
        if slack == 0:
            return

        # fill A and c with 0's
        self.A = np.hstack((self.A, np.zeros((self.m, slack))))
        self.c = np.concatenate((self.c, np.zeros(slack)))

        # add slack variables
        s = self.n
        for i in range(self.m):
            if self.R[i] == '<=':
                self.A[i, s] = 1
                self.R[i] = '=='
                # save the index of those that will be part of the base
                self.B_i[i] = s
                s += 1
            elif self.R[i] == '>=':
                self.A[i, s] = -1
                s += 1
                self.R[i] = '=='

    def __artificial_variables(self):

        # find (line) index of variables to be added
        need_var = [ i for i in range(self.m) if i not in self.B_i.keys() ]

        # fill with zeros -> m lines and as much columns as varibles needed
        self.A = np.hstack((self.A, np.zeros((self.m, len(need_var)))))

        # place 1 at the right spot and update the base index
        col = self.n
        for line in need_var:
            self.A[line, col] = 1
            self.B_i[line] = col
            col += 1

    def solve(self):
        self.__standart_form()
        # n gets a new value (old_n + slack_variables)
        m, self.n = np.shape(self.A)

        # go to phase I if there are not enough variables in the base
        if self.m > len(self.B_i):
            print "Add artificial variables"
            self.__artificial_variables()
            # n gets a new value (old_n + artificial_variables)
            m, self.n = np.shape(self.A)

        # go to phase II

    def print_problem(self):
        print '#'*30
        print 'Problem: ', self.name
        print '\nObjective: ', self.obj
        print '\nCosts: ', self.c
        print '\nConstraints: ', self.R
        print '\nA:\n', self.A
        print '\nb: ', self.b
        print '#'*30



def read_json(input_file):
    with open(input_file, 'r') as f:
        data = json.load(f)

    return data



def main(input_file):
    model = Model(read_json(input_file), debug=True)

    model.solve()
    model.print_problem()
    #model.print_solution()




if __name__ == '__main__':
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        print "\nInput file was expected.\nExiting...\n"
        exit(1)

