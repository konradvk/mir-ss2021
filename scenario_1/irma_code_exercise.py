import csv
import os
from os.path import join

def csv_to_dict(file_path, delimiter):
    """
    Function to read in a csv file and create a dict based on the first two columns.

    Parameters
    ----------
    - file_path : string
        The filepath to the CSV file.
    
    Returns
    -------
    - csv_dict : dict
        The dict created from the CSV file.

    Tipps
    -------
    - Read in the CSV file from the path
    - For each row, add an entry to a dict (first column is key, second column is value)
    - Return the dict
    """
    csv_dict = {}
    with open(file_path) as f:
        # initialize the CSV reader
        reader = csv.reader(f, delimiter=delimiter)

        # loop over the rows in the csv file
        for row in reader:
            # exclude empty rows and rows with single semicolons
            if row and row[0] is not '':
                # add to dictionary; Key: first column, Item: second column
                csv_dict[row[0]] = row[1]
    f.close()
    return csv_dict

class IRMA:
    """
    Class to retrieve the IRMA code and information for a given file.
    """
    labels_long = ["Technical code for imaging modality", "Directional code for imaging orientation", "Anatomical code for body region examined", "Biological code for system examined"]
    labels_short = ["Imaging modality", "Imaging orientation", "Body region", "System"]

    def __init__(self, dir_path= os.path.join("static", "codes")):
        """
        Constructor of an IRMA element.

        Parameters
        ----------
        - dir_path : string
            The path where the irma data is. There should be a "A.csv", "B.csv", "C.csv", "D.csv" and "image_codes.csv" file in the directory.

        Tipps
        -------
        - Create a dict for part A, B, C, and D of the IRMA code (user csv_to_dict(file_path))
        - Save the dicts (list) as class variable
        - Save "image_codes.csv" as dict in a class variable
        """
        
        self.codes_dict = csv_to_dict(join(dir_path, 'codes.csv'), delimiter=',')
        A_dict = csv_to_dict(join(dir_path, 'A.csv'), delimiter=';')
        B_dict = csv_to_dict(join(dir_path, 'B.csv'), delimiter=';')
        C_dict = csv_to_dict(join(dir_path, 'C.csv'), delimiter=';')
        D_dict = csv_to_dict(join(dir_path, 'D.csv'), delimiter=';')

        self.irma_parts = [A_dict, B_dict, C_dict, D_dict]


    def get_irma(self, image_names):
        """
        Function to retrieve irma codes for given image names.

        Parameters
        ----------
        - image_names : list
            List of image names.

        Returns
        -------
        - irma codes : list
            Retrieved irma code for each image in 'image_list'

        Tipps
        -------
        - Remove file extension and path from all names in image_names. Names should be in format like first column of 'image_codes.csv'
        - Use self.image_dict to convert names to codes. ('None' if no associated code can be found)
        - Return the list of codes
        """
        image_names = [x[:-4] for x in image_names]
        return [self.codes_dict[x] if x in self.codes_dict else None for x in image_names]


    def decode_as_dict(self, code):
        """
        Function to decode an irma code to a dict.

        Parameters
        ----------
        - code : str
            String to decode.

        Returns
        -------
        - decoded : dict

        Tipps
        -------
        - Make use of 'labels_short'
        - Possible solution: {'Imaging modality': ['x-ray', 'plain radiography', 'analog', 'overview image'], ...}
        - Solution can look different
        """
        # Split code into its 4 parts 
        split_codes = code.split('-')
        decoded = {}
        for i,category in enumerate(self.labels_short):
            # get code part for the current category
            current_code = split_codes[i]
            # Search all substrings of the current code in the respective category-dictionary (only include if found in dictionary)
            description = [self.irma_parts[i][current_code[:end+1]] for end,_ in enumerate(current_code) if current_code[:end+1] in self.irma_parts[i]]
            # description = [self.irma_parts[i][current_code[:end]] for end in range(1,5) if current_code[:end] in self.irma_parts[i]]
            # description = [ current_code[:end+1] for end,_ in enumerate(current_code)]
            decoded[category] = description

        return decoded

    def decode_as_str(self, code):
        """
        Function to decode an irma code to a str.

        Parameters
        ----------
        - code : str
            String to decode.

        Returns
        -------
        - decoded : str
            List of decoded strings.

        Tipps
        -------
        - Make use of 'decode_as_dict'
        - Possible solution: ['Imaging modality: x-ray, plain radiography, analog, overview image', 'Imaging orientation: coronal, anteroposterior (AP, coronal), supine', 'Body region: abdomen, unspecified', 'System: uropoietic system, unspecified']
        - Solution can look different -> FLASK will use this representation to visualize the information on the webpage.
        """
        # get dictionary of code entries
        decoded_dict = self.decode_as_dict(code)
        decoded = []
        for key, values in decoded_dict.items():
            # first element is the key, join the values with an ',' as delimiter
            category_string = key + ': ' + ', '.join(values)
            decoded.append(category_string)

        return decoded

if __name__ == '__main__':
    image_names = ["1880.png", "3403.png"]
    # image_names = ["3403.png"]

    irma = IRMA()

    codes = irma.get_irma(image_names)
    print("Codes: ", codes)

    if codes is not None:
        for code in codes:
            if code is None:
                print('No matching IRMA code was found for this image!')
            else:
                print("Dict: \n{}\n\n".format(irma.decode_as_dict(code)))
                print("String: \n{}\n\n".format(irma.decode_as_str(code)))

    '''
    Result could look like this:


    Codes:  ['1121-127-700-500']
    Dict:
    {'Imaging modality': ['x-ray', 'plain radiography', 'analog', 'overview image'], 'Imaging orientation': ['coronal', 'anteroposterior (AP, coronal)', 'supine'], 'Body region': ['abdomen', 'unspecified'], 'System': ['uropoietic system', 'unspecified']}


    String:
    ['Imaging modality: x-ray, plain radiography, analog, overview image', 'Imaging orientation: coronal, anteroposterior (AP, coronal), supine', 'Body region: abdomen, unspecified', 'System: uropoietic system, unspecified']
    '''