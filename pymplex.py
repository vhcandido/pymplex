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

    def __iterate(self):
        B_i = self.B_i
        N_i = self.N_i
        B_inv = np.linalg.inv(self.A[:, B_i]) # invert base
        b = self.b[np.newaxis].T # transpose b array
        x_B = B_inv * b
        x_B = np.array(x_B)[:, 0] # transform matrix into array

        # current solution
        c_B = self.c[B_i]
        self.function = c_B * x_B

        # lambda - simplex multiplier
        lam = c_B * B_inv

        # new relative costs
        self.c[N_i] = self.c[N_i] - lam * A[:, N_i]

        # if costs are all negative then the solution was found
        if (self.c[N_i] >= 0).all():
            print "Solution found"
            exit()

        # find who is going to enter the base
        goes_in_i = np.argmin(self.c[N_i])
        goes_in = N_i[goes_in_i]

        # calculate directions
        y = B_inv * A[:, goes_in]
        # if it's not positive then the problem has no limited solution
        if (y <= 0).all():
            print "Unbounded optimal solution"
            exit()

        # find who is going to leave the base
        y = np.array(y)[:, 0]

        #eps = []
        #for i in range(m):
        #    if y[i] > 0:
        #        eps.append(x[i] / y[i])
        eps = np.apply_along_axis(lambda(x,y): x/y if y>0 else np.inf, 1, zip(x_B, y))
        goes_out_i = np.argmin(e)
        goes_out = B_i[goes_out_i]

        # update base
        B_i[goes_out_i], N_i[goes_in_i] = N_i[goes_in_i], B_i[goes_out_i]
        self.x[B_i] = x_B
        self.B_i = B_i
        self.N_i = N_i
        self.iteration += 1

    def solve(self):
        self.__standart_form()
        # n gets a new value (old_n + slack_variables)
        m, self.n = np.shape(self.A)

        # go to phase I if there are not enough variables in the base
        if self.m > len(self.B_i):
            self.__artificial_variables()
            # n gets a new value (old_n + artificial_variables)
            m, self.n = np.shape(self.A)

        # go to phase II
        self.__iterate()

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

