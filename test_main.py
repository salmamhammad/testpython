import numpy as np
import pytest
from unittest.mock import MagicMock


class MatrixCalc:
    def matAEntry(self):
        # Mock this method for testing
        pass
    def precisionVal(self):
        # Mock this method for testing
        pass
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
        except Exception as err:
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


class TestGenerateMatA:
    
    def test_generateMatA_valid_input(self,monkeypatch):
       # Mocking GUI elements
      matA_entry_mock = MagicMock()
      matA_entry_mock.toPlainText.return_value = "1 2\n3 4"  # Example matrix input

      precision_val_mock = MagicMock()
      precision_val_mock.currentText.return_value = "2"  # Example precision value

    # Applying the mocks using monkeypatch
      monkeypatch.setattr(MatrixCalc,'matAEntry', matA_entry_mock)
      monkeypatch.setattr(MatrixCalc,'precisionVal', precision_val_mock)

    # Creating an instance of MatrixCalc
      obj = MatrixCalc()

    # Call the method
      result = obj.generateMatA()

    # Assertions
      expected_result = np.array([[1, 2], [3, 4]])
      assert np.array_equal(result, expected_result)
     
 
    def test_generateMatA_invalid_input(self,monkeypatch):
 
    # Mocking GUI elements
      matA_entry_mock = MagicMock()
      matA_entry_mock.toPlainText.return_value = "1 2\n3"  # Invalid matrix input

      precision_val_mock = MagicMock()
      precision_val_mock.currentText.return_value = "2"

    # Applying the mocks using monkeypatch
      monkeypatch.setattr(MatrixCalc,'matAEntry', matA_entry_mock)
      monkeypatch.setattr(MatrixCalc,'precisionVal', precision_val_mock)

    # Creating an instance of MatrixCalc
      obj = MatrixCalc()

    # Call the method
      result = obj.generateMatA()

    # Assertions
      expected_result = 'Rows or Columns of Matrix A do not Match'
      assert result == expected_result
 
    def test_check_file_existence(self, tmp_path,monkeypatch):
        matA_entry_mock = MagicMock()
        matA_entry_mock.toPlainText.return_value = "1 2\n3 4"  # Example matrix input

        precision_val_mock = MagicMock()
        precision_val_mock.currentText.return_value = "2"  # Example precision value

        # Applying the mocks using monkeypatch
        monkeypatch.setattr(MatrixCalc,'matAEntry', matA_entry_mock)
        monkeypatch.setattr(MatrixCalc,'precisionVal', precision_val_mock)
        # Create an instance of YourClass
        your_instance = MatrixCalc()

        # Call the method that writes to the file
        your_instance.generateMatA()

        # Get the path to the file
        file_path = tmp_path / 'E:/python/matrix-calculator-python/v1/matA.txt'

        # Check if the file exists
        assert  file_path.exists()

    def test_check_file_Hot_existence(self, tmp_path,monkeypatch):
        matA_entry_mock = MagicMock()
        matA_entry_mock.toPlainText.return_value = "1 2\n3 4"  # Example matrix input

        precision_val_mock = MagicMock()
        precision_val_mock.currentText.return_value = "2"  # Example precision value

        # Applying the mocks using monkeypatch
        monkeypatch.setattr(MatrixCalc,'matAEntry', matA_entry_mock)
        monkeypatch.setattr(MatrixCalc,'precisionVal', precision_val_mock)
        # Create an instance of YourClass
        your_instance = MatrixCalc()

        # Call the method that writes to the file
        your_instance.generateMatA()

        # Get the path to the file
        file_path = tmp_path / 'E:/python/matrix-calculator-python/v1/matA4.txt'

        # Check if the file exists
        assert  file_path.exists()
class TestmatA:        
  @pytest.fixture
  def generateMatA_valid_input(self,monkeypatch):

    # Mocking GUI elements
    matA_entry_mock = MagicMock()
    matA_entry_mock.toPlainText.return_value = "1 2\n3 4"  # Example matrix input

    precision_val_mock = MagicMock()
    precision_val_mock.currentText.return_value = "2"  # Example precision value

    # Applying the mocks using monkeypatch
    monkeypatch.setattr(MatrixCalc,'matAEntry', matA_entry_mock)
    monkeypatch.setattr(MatrixCalc,'precisionVal', precision_val_mock)

    # Creating an instance of MatrixCalc
    obj = MatrixCalc()

    # Call the method
    result = obj.generateMatA()

    # Assertions
    return result
  @pytest.fixture
  def generateMatA_invalid_input(self,monkeypatch):

    # Mocking GUI elements
    matA_entry_mock = MagicMock()
    matA_entry_mock.toPlainText.return_value = "1 2\n3"  # Invalid matrix input

    precision_val_mock = MagicMock()
    precision_val_mock.currentText.return_value = "2"

    # Applying the mocks using monkeypatch
    monkeypatch.setattr(MatrixCalc,'matAEntry', matA_entry_mock)
    monkeypatch.setattr(MatrixCalc,'precisionVal', precision_val_mock)

    # Creating an instance of MatrixCalc
    obj = MatrixCalc()

    # Call the method
    result = obj.generateMatA()

    return result

  def test_matA_func(self,generateMatA_valid_input):

    # Call the method
    result=str( generateMatA_valid_input)

    # Assuming generateMatA() returns some matrix, you might want to mock this if it's not deterministic
    expected_result ='[[1. 2.]\n [3. 4.]]'  # Example expected output
    
    # Assert the result
    assert result == expected_result

  def test_matA_func_invalid_value(self,generateMatA_invalid_input):
    # Call the method
    result=str( generateMatA_invalid_input)

    # Assuming generateMatA() returns some matrix, you might want to mock this if it's not deterministic
    expected_result ='Rows or Columns of Matrix A do not Match'  # Example expected output
    
    # Assert the result
    assert result == expected_result

class TestinvA:        
  def test_invA_func_valid_matrix(self,mocker):
    # Mock generateMatA method to return a valid matrix
    mocker.patch.object(MatrixCalc, 'generateMatA', return_value=np.array([[1, 2], [3, 4]]))
    
    obj = MatrixCalc()
    result = obj.invA_func()
    expected_result = 'Inverse of Matrix A=\n [[-2.   1. ]\n [ 1.5 -0.5]]'
    assert result == expected_result

  def test_invA_func_invalid_matrix(self,mocker):
    # Mock generateMatA method to return a string (indicating an error)
    mocker.patch.object(MatrixCalc, 'generateMatA', return_value='Error: Matrix A is not valid')
    
    obj = MatrixCalc()
    result = obj.invA_func()
    expected_result = 'Rows or Columns of Matrix A do not Match'
    assert result == expected_result

  def test_invA_func_non_invertible_matrix(self,mocker):
    # Mock generateMatA method to return a non-invertible matrix
    mocker.patch.object(MatrixCalc, 'generateMatA', return_value=np.array([[1, 2], [2, 4]]))
    
    obj = MatrixCalc()
    result = obj.invA_func()
    expected_result = 'Inverse of Matrix A \nSingular matrix'
    assert result == expected_result

class TesttransA:
    def test_transA_valid_input(self, mocker):
        # Mock the behavior of generateMatA to return a valid numpy array
        mocker.patch.object(MatrixCalc, 'generateMatA', return_value=np.array([[1, 2], [3, 4]]))
        obj = MatrixCalc()
        # Call the method
        result = obj.transA_func()

        # Assertions
        expected = '''Transpose of Matrix A=
[[1 3]
 [2 4]]'''
        assert result == expected

    def test_transA_invalid_input(self, mocker):
        # Mock the behavior of generateMatA to return an error message
        mocker.patch.object(MatrixCalc, 'generateMatA', return_value='Rows or Columns of Matrix A do not Match')
        obj = MatrixCalc()
      
        # Call the method
        result = obj.transA_func()

        # Assertions
        assert result == 'Rows or Columns of Matrix A do not Match'

    def test_transA_ExceptionError(self, mocker):
        # Mock the behavior of generateMatA to raise Exception
        mocker.patch.object(MatrixCalc, 'generateMatA', return_value=  Exception("Some exception occurred"))
        obj = MatrixCalc()
      
        # Call the method
        result = obj.transA_func()

        # Assertions
        assert result == "Transpose of Matrix A\n'Exception' object has no attribute 'T'"

class TestdetA:


    def test_detA_valid_input(self, mocker):
        # Mock the behavior of generateMatA to return a valid numpy array
       
        mocker.patch.object(MatrixCalc, 'generateMatA', return_value=np.array([[1, 0], [1, 0]]))
       
        # Mock the precisionVal.currentText() method
        mocker.patch.object(MatrixCalc, 'precisionVal', return_value= "2")

        obj = MatrixCalc()
        # Call the method
        result = obj.detA_func()

        # Assertions
        expected = '''Determinant of Matrix A=\n\t0.0'''
        assert result == expected

    def test_detA_invalid_input(self, mocker):
        # Mock the behavior of generateMatA to return an error message
        mocker.patch.object(MatrixCalc, 'generateMatA', return_value=  'Rows or Columns of Matrix A do not Match')

        obj = MatrixCalc()
        # Call the method
        result = obj.detA_func()

        # Assertions
        assert result == 'Rows or Columns of Matrix A do not Match'


    def test_detA_LinAlgError(self, mocker):
        # Mock the behavior of generateMatA to return a valid numpy array
        
        mocker.patch.object(MatrixCalc, 'generateMatA', return_value=  np.array([[1, 2], [3, 4]]))

        # Mock the behavior of np.linalg.det to raise LinAlgError
        mocker.patch('numpy.linalg.det', side_effect=np.linalg.LinAlgError("Singular matrix"))
        
         # Mock the precisionVal.currentText() method
        mocker.patch.object(MatrixCalc, 'precisionVal', return_value= "2")

        obj = MatrixCalc()

        # Call the method
        result = obj.detA_func()

        # Assertions
        expected = 'Determinant of Matrix A\nSingular matrix'
        assert result == expected


class TestArrMulmat:
  

    def test_arrMulmat_valid_input(self, mocker):
        # Mock the behavior of generateMatA and generateMatB to return valid numpy arrays
        mocker.patch.object(MatrixCalc, 'generateMatA', return_value=  np.array([[1, 2], [3, 4]]))
        mocker.patch.object(MatrixCalc, 'generateMatB', return_value=  np.array([[5, 6], [7, 8]]))
 
        obj = MatrixCalc()

        # Call the method
        result = obj.arrMulmat_func()

        # Assertions
        expected = '''Element-wise Multiplication of  A and B= 
[[ 5 12]
 [21 32]]'''
        assert result == expected

    def test_arrMulmat_invalid_input(self, mocker):
        # Mock the behavior of generateMatA to return an error message
        mocker.patch.object(MatrixCalc, 'generateMatA', return_value=  'Rows or Columns of Matrix A do not Match')
        mocker.patch.object(MatrixCalc, 'generateMatB', return_value=  np.array([[5, 6], [7, 8]]))
 
        obj = MatrixCalc()
    
        # Call the method
        result = obj.arrMulmat_func()

        # Assertions
        assert result == 'Rows or Columns of Matrix A or B do not Match'

    def test_arrMulmat_exception(self, mocker):
        # Mock the behavior of generateMatA to raise a specific exception
        mocker.patch.object(MatrixCalc, 'generateMatA', return_value= FloatingPointError("Floating Point Error"))
        mocker.patch.object(MatrixCalc, 'generateMatB', return_value=  np.array([[5, 6], [7, 8]]))
        
        obj = MatrixCalc()

        # Call the method
        result = obj.arrMulmat_func()

        # Assertions
        assert result == "Element-wise Multiplication of Matrix A and B\nunsupported operand type(s) for *: 'FloatingPointError' and 'int'"



    def test_arrMulmat_exception_multiplication(self, mocker):
        # Mock the behavior of np.multiply to raise a specific exception
        mocker.patch.object(MatrixCalc, 'generateMatA', return_value= np.array([[1, 2], [3, 4]]))
        mocker.patch.object(MatrixCalc, 'generateMatB', return_value=  np.array([[5, 6], [7, 8]]))
        
        obj = MatrixCalc()

        mocker.patch('numpy.multiply', side_effect=ValueError("Invalid shapes for multiplication"))
        

        # Call the method
        result = obj.arrMulmat_func()

        # Assertions
        assert result == 'Element-wise Multiplication of Matrix A and B\nInvalid shapes for multiplication'
