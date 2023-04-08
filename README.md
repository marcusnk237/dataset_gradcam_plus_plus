<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a name="readme-top"></a>
<br />
<div align="center">
  <h1 align="center">Dataset level explainability for time series classification</h1>

  <p align="center">
one of the first implementation of Grad-CAM ++ for time series / 1d signal. The module  gives a visual explainations of the decisions made by Deep Learning model (Especially for classification problems) and helps to understand how the model works and assert the model results.
    <br />
    <a href="https://github.com/othneildrew/Best-README-Template"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/marcusnk237/dataset_gradcam_plus_plus/issues">Report Bug</a>
    ·
    <a href="https://github.com/marcusnk237/dataset_gradcam_plus_plus/issues">Request Feature</a>
  </p>
</div>
<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

[![Product Name Screen Shot][product-screenshot]](https://example.com)

One of the main challenges in artificial intelligence for the researchers is to understands how model predictions works.
Many contributions has been made, especially GRAD-CAM++.
Grad-CAM++ give a visual representation of the keys features responsible of the classification, and give human-level understanding of the model prediction. 
Despite its advantages, GRAD-CAM++ have drawbacks:

* GRAD-CAM ++ is not initially design for time series

* GRAD-CAM ++ works only for local classification. It can give any information about key features responsible of the classification at a dataset-level.

Our library not only give a GRAD-CAM ++ visualisation for time series, but also give keys feature importances at a dataset level

<p align="right">(<a href="#readme-top">back to top</a>)</p>



### Built With

[![Python][Python]][Python-url]
[![Tensorflow][Tensorflow]][Tensorflow-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->
## Getting Started

### Prerequisites
[![Tensorflow][Tensorflow]][Tensorflow-url]
[![Opencv][Opencv]][Opencv-url]
[![Pandas][Pandas]][Pandas-url]
[![Matplotlib][Matplotlib]][Matplotlib]
[![Numpy][Numpy]][Numpy-url]

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/marcusnk237/dataset_gradcam_plus_plus.git
   ```
2. Install the library
   ```python3
   setup.py install
   ```
<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Usage

<!-- LICENSE -->
## License

Distributed under the GNU License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact
* [![LinkedIn][linkedin-shield]][linkedin-url]

* Project Link: [https://github.com/marcusnk237/dataset_gradcam_plus_plus](https://github.com/marcusnk237/dataset_gradcam_plus_plus)
<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

The authors of the original article about GRAD-CAM++
* [Aditya Chattopadhay; Anirban Sarkar; Prantik Howlader; Vineeth N Balasubramanian : Grad-CAM++: Generalized Gradient-Based Visual Explanations for Deep Convolutional Networks](https://doi.org/10.1109/WACV.2018.00097)
<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]: https://github.com/marcusnk237/dataset_gradcam_plus_plus/blob/main/LICENSE
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/marc-junior-nkengue/
[product-screenshot]: images/screenshot.png

[Opencv]:https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white
[Opencv-url]:https://pypi.org/project/opencv-python/
[Pandas]:https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white
[Pandas-url]:https://pandas.pydata.org/
[Matplotlib]:https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black
[Matplotlib-url]:https://matplotlib.org/
[NumPy]: https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white
[Numpy-url]:https://numpy.org/
[Python]: https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=yellow
[Python-url]: https://www.python.org/
[Tensorflow]: https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white 
[Tensorflow-url]:  https://www.tensorflow.org/

