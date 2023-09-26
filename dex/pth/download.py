import gdown

url1 = "https://drive.google.com/uc?id=1ancMcq5b_0nppdIAav1UgBxdiaMnnQPe"
url2 = "https://drive.google.com/uc?id=1KtzJBcGPMwjYdxbJkfEeRFzxNMaTVykH"

gdown.download(url1, "gender_sd.pth", quiet=False)
gdown.download(url2, "age_sd.pth", quiet=False)