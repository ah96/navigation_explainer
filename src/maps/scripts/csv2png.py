import csv
from PIL import Image


def convert_csv_to_png(csv_file, png_file):
    # Read pixel values from CSV
    with open(csv_file, 'r') as file:
        reader = csv.reader(file, delimiter=',')
        data = list(reader)[1:]

    for i in range(0, len(data)):
        data[i].extend([205] * (1984-1024))    

    height = len(data)
    width = len(data[0])

    print(len(data))
    print(len(data[0]))

    # Create a new image from pixel values
    image = Image.new('L', (width, height))
    pixels = [int(pixel) for row in data for pixel in row]
    #print(pixels[-100:-1])
    image.putdata(pixels)

    # Save the image as PNG
    image.save(png_file)


# Usage example
csv_file = 'mapnew.csv'
png_file = 'mapnew.png'

convert_csv_to_png(csv_file, png_file)
