## ����Ҫ��
1. ����face detection�� face alignment ����������֮��Ĺ��ԣ�ͨ������������һ��CNNͬʱ�������¡�
2. ����,������Ƶ�CNN��3��stage���ܿ��ٲ�����ѡ���ڵ�ǳ��CNN --feed--> ���޳������������ݵ��Ը��ӵ�CNN --feed--> �ܱ��landmark����CNN��������������ߺܶࡣ
3. Online hard sample mining�����򴫲�ֻ���������ߵ�70%���������ݶ��½�����������ܡ�
4. �ܵ�ѧϰ��ʧ����������Ҫ�Զ�̬���㡣���ڲ�ͬstages ��� face/non-face classification + bounding box regression + facial landmark localization�����ߵ���ʧ ��
5. �Ż�filters��ʹ�ø�С�������Ĺ�������

## �������ʵ�֣�
0. ���ȸ������Ľ���P R O ��������ģ��

1. ����ͼ�������
��ԭͼ�ߴ練�������������ӣ��Ա�õ������boundingbox��

2. ���õ���ͬ�ߴ��ͼƬ���뵽PNet����ǰ������
����һ����N�����ź��ͼƬ���ŵ�PNet���磬�õ�boundingboxex

3. ��RNetɸѡ��ѡ��boundingbox
���Ȱ���ÿ����ѡ��������Ϣ��ԭͼ�л�ȡ���ݣ����ҽ����е�����imResample����24��24���ĳߴ磬�������뵽RNet��������ǰ�����㡣ɸѡ��һ���ֺ�ѡ��

4. ONet��RNet���������һ�������õ�Ψһ�ĺ�ѡ��
���Ȱ���ÿ����ѡ��������Ϣ��ԭͼ�л�ȡ���ݣ����ҽ����е�����imResample����48��48���ĳߴ磬�������뵽ONet��������ǰ�����㡣ɸѡ��һ���ֺ�ѡ��,�õ�Ψһ��������
����ONet���������������ֵ����������մ����Ӧ��ϵ���õ�ԭͼ��Ӧ������ؼ��������ֵ��

##Train
��������IoU >= 0.65
��������IoU < 0.3
����(part)������0.65 > IoU >= 0.4
landmark����

������������face classification tasks
��������part��������bounding box regression
landmark���� ����facial landmark localization
landmark faces are used for facial landmark localization. 

## Demo usage
��˵����ʹ��python + tensorflow ʵ�ֱȽϷ��㣬�ʴ˰汾ʹ��pythonд�ģ����ڿɸ�ΪC++��

1. �tensorflow, �ο�https://www.tensorflow.org/install
2. ��װpython��: opencv, numpy
3. python ./facedetect_mtcnn.py --input ./test.jpg --output  new.jpg

## Results
�����ǰĿ¼�µ�test.jpg �� new.jpg
