#
import setuptools
from setuptools import setup


metadata = {'name': 'shock_pods',
            'maintainer': 'Edward Azizov',
            'maintainer_email': 'edazizovv@gmail.com',
            'description': 'NLP package',
            'license': 'MIT',
            'url': 'https://github.com/edazizovv/shock_pods',
            'download_url': 'https://github.com/edazizovv/shock_pods',
            'packages': setuptools.find_packages(),
            'include_package_data': True,
            'version': '0.1.4',
            'long_description': '',
            'python_requires': '==3.9.*',
            'install_requires': []}

setup(**metadata)
