from setuptools import find_packages, setup 

setup(
    name='Medical Chatbot',
    version='0.0.1',
    author='ansh agarwal',
    author_email='agarwalansh0511@gmail.com',
    install_requires=["pinecone", "langchain", "flask", "python-dotenv", "PyPDF2"],
    packages=find_packages()
)