# ML_Pythion_TitanicWithPandasAndTensorflow

google colab에 titanic data를 받는 법

`kaggle.json` 파일을 `/content`에 업로드

```bash
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle
!chmod 600 ~/.kaggle/kaggle.json
!kaggle competitions download -c titanic
!unzip titanic.zip
```
