#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 21:37:30 2020

@author: Joe Yapzor
"""


import numpy as np
import scipy.linalg as LA
from UI import ui as uiw


class MatrixCalc:
  
  
    def generateMatB(self):
        matB = self.matBEntry.toPlainText()
        precisionVal = int(self.precisionVal.currentText())
        np.set_printoptions(precision=precisionVal)
        with open('matB.txt', 'w', encoding="utf-8") as f:
            f.write(matB)
            f.close()
        try:
            B = np.genfromtxt('matB.txt', delimiter=' ')
            return B
        except ValueError:
            return 'Rows or Columns of Matrix B do not Match'

    def generateMatA(self):
        matA = self.matAEntry.toPlainText()
        precisionVal = int(self.precisionVal.currentText())
        np.set_printoptions(precision=precisionVal)
        with open('matA.txt', 'w', encoding="utf-8") as f:
            f.write(matA)
            f.close()
        try:
            A = np.genfromtxt('matA.txt', delimiter=' ')
            return A
        except ValueError:
            return 'Rows or Columns of Matrix A do not Match'


    def matA_func(self):

        A = self.generateMatA()
        A = f'''Matrix A=\n{A}'''
        return str(A)

    def invA_func(self):
        A = self.generateMatA()

        if isinstance(A, str):
            return 'Rows or Columns of Matrix A do not Match'
        try:
            invA = np.linalg.inv(A)
            invA = f'''Inverse of Matrix A=\n {invA}'''
            return invA
        except np.linalg.LinAlgError as err:
            return f'''Inverse of Matrix A \n{err}'''

    def transA_func(self):
        A = self.generateMatA()
        if isinstance(A, str):
            return 'Rows or Columns of Matrix A do not Match'
        try:
            transA = A.T
            transA = f'''Transpose of Matrix A=\n{transA}'''
            return transA
        except FloatingPointError as err:
            return f'''Transpose of Matrix A\n{err}'''

 

    def detA_func(self):
        A = self.generateMatA()
        if isinstance(A, str):
            return 'Rows or Columns of Matrix A do not Match'
        try:
            precisionVal = int(self.precisionVal.currentText())
            detA = np.linalg.det(A)
            detA = round(detA, precisionVal)
            detA = f'''Determinant of Matrix A=\n\t{detA}'''
            return detA
        except np.linalg.LinAlgError as err:
            return f'''Determinant of Matrix A\n{err}'''

    def traceA_func(self):
        A = self.generateMatA()
        if isinstance(A, str):
            return 'Rows or Columns of Matrix A do not Match'
        try:
            precisionVal = int(self.precisionVal.currentText())
            traceA = np.trace(A)
            traceA = round(traceA, precisionVal)
            traceA = f'''Trace of Matrix A=\n\t{traceA}'''
            return traceA
        except FloatingPointError as err:
            return f'''Trace of Matrix A\n{err}'''
        
    def arrMulmat_func(self):
        A = self.generateMatA()
        B = self.generateMatB()
        if isinstance(B, str) or isinstance(A, str):
            return 'Rows or Columns of Matrix A or B do not Match'
        try:
            mulMat = np.multiply(A, B)
            mulMat = f'''Element-wise Multiplication of  A and B= \n{mulMat}'''
            return mulMat
        except Exception as err:
            return f'''Element-wise Multiplication of Matrix A and B\n{err}'''
