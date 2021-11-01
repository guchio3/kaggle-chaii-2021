import os


def main():
    os.system("git pull")
    os.system("mkdir ./temp_kaggle_upload")
    os.system("cp -r configs ./temp_kaggle_upload")
    os.system("cp -r src ./temp_kaggle_upload")
    os.system("cp -r dataset-metadata.json ./temp_kaggle_upload")
    os.system('kaggle datasets version -p ./temp_kaggle_upload -r tar -m "temp" -d')

    # remove
    os.system("rm -r ./temp_kaggle_upload")


if __name__ == "__main__":
    main()
