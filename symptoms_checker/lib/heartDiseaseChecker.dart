import 'dart:async';

import 'package:custom_radio_grouped_button/CustomButtons/ButtonTextStyle.dart';
import 'package:custom_radio_grouped_button/CustomButtons/CustomRadioButton.dart';
import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_easyloading/flutter_easyloading.dart';
import 'package:pytorch_mobile/enums/dtype.dart';
import 'package:pytorch_mobile/model.dart';
import 'package:pytorch_mobile/pytorch_mobile.dart';
import 'package:symptoms_checker/main.dart';

class HeartDiseaseChecker extends StatefulWidget{
  HeartDiseaseChecker({Key key}) : super(key: key);

  @override
  _HeartDiseaseCheckerState createState() => _HeartDiseaseCheckerState();
}

class _HeartDiseaseCheckerState extends State<HeartDiseaseChecker> {
  int gender = 0;
  double age = 0;
  int chestPain = 0;
  double restingBloodPressure = 0;
  double cholesterol = 0;
  int fbs = 0;
  int rer = 0;
  double maxHeartRate = 0;
  int exang = 0;
  double oldpeak = 0;
  int slope = 0;
  int noOfVessels = 0;
  int thal = 0;

  List<double> input;
  Timer _timer;

  Model _model;
  List _prediction;
  bool result;

  @override
  void initState() {
    super.initState();
    loadModel();
  }

  Future loadModel() async {
    String _modelPath = "assets/models/heartDiseaseTest.pt";
    try {
      _model = await PyTorchMobile.loadModel(_modelPath);
    } on PlatformException {
      print("only supported for android so far");
    }
  }


  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Symptoms Checker'),
        backgroundColor: Colors.indigo[200],
        bottomOpacity: 0.5,
        elevation: 0.5,
      ),
      body: Container(
          width: MediaQuery.of(context).size.width,
          height: MediaQuery.of(context).size.height,
          padding: EdgeInsets.all(20),
          child: SingleChildScrollView(
            child: Column(
              children: [

                Container(
                  width: MediaQuery.of(context).size.width,
                  padding: EdgeInsets.all(10),
                  alignment: Alignment.center,
                  color: Colors.indigo[200],
                  child: Text(
                    'HEART DISEASE CHECKER',
                    style: TextStyle(
                        fontWeight: FontWeight.bold,
                        fontSize: 20,
                        color: Colors.white
                    ),
                  ),
                ),

                Padding(
                  padding: EdgeInsets.only(top: 20),
                  child: Text(
                    'Gender?',
                    style: TextStyle(
                        fontWeight: FontWeight.bold,
                        fontSize: 15
                    ),
                  ),
                ),

                Padding(
                  padding: EdgeInsets.only(top: 2),
                  child: CustomRadioButton(
                    elevation: 10,
                    absoluteZeroSpacing: false,
                    unSelectedColor: Colors.white,
                    buttonLables: [
                      'Male',
                      'Female',
                    ],
                    buttonValues: [
                      1,
                      0,
                    ],
                    buttonTextStyle: ButtonTextStyle(
                        selectedColor: Colors.white,
                        unSelectedColor: Colors.black,
                        textStyle: TextStyle(fontSize: 12, fontWeight: FontWeight.bold)),
                    radioButtonValue: (value) {
                      setState(() {
                        gender = value;
                      });
                      print(value);
                    },
                    selectedColor: Colors.green[700],
                    width: 80,
                    enableButtonWrap: true,
                  ),
                ),

                Padding(
                  padding: EdgeInsets.only(top: 20),
                  child: Text(
                    'Age?',
                    style: TextStyle(
                        fontWeight: FontWeight.bold,
                        fontSize: 15
                    ),
                  ),
                ),

                Container(
                  margin: EdgeInsets.only(left: 20, right: 20, top: 5),
                  height: 48,
                  child: TextFormField(
                    validator: (value) {
                      print(value);
                      if (value.isEmpty) {
                        return 'This field cannot be empty';
                      }
                      return null;
                    },
                    keyboardType: TextInputType.number,
                    textAlign: TextAlign.center,
                    decoration: InputDecoration(
                      enabledBorder: OutlineInputBorder(
                          borderRadius: BorderRadius.all(Radius.circular(10)),
                          borderSide: BorderSide(color: Colors.grey)
                      ),
                      focusedBorder: OutlineInputBorder(
                          borderRadius: BorderRadius.all(Radius.circular(10)),
                          borderSide: BorderSide(color: Colors.grey)
                      ),
                      filled: true,
                      fillColor: Colors.white,
                    ),
                    onChanged: (value) {
                      age = double.parse(value);
                      print(age);
                    },
                  ),

                ),

                Padding(
                  padding: EdgeInsets.only(top: 20),
                  child: Text(
                    'Chest Pain?',
                    style: TextStyle(
                        fontWeight: FontWeight.bold,
                        fontSize: 15
                    ),
                  ),
                ),

                Padding(
                  padding: EdgeInsets.only(top: 2),
                  child: CustomRadioButton(
                    elevation: 10,
                    absoluteZeroSpacing: false,
                    unSelectedColor: Colors.white,
                    buttonLables: [
                      'None',
                      'Low',
                      'Moderate',
                      'High'
                    ],
                    buttonValues: [
                      0,
                      1,
                      2,
                      3
                    ],
                    buttonTextStyle: ButtonTextStyle(
                        selectedColor: Colors.white,
                        unSelectedColor: Colors.black,
                        textStyle: TextStyle(fontSize: 12, fontWeight: FontWeight.bold)),
                    radioButtonValue: (value) {
                      setState(() {
                        chestPain = value;
                      });
                      print(value);
                    },
                    selectedColor: Colors.green[700],
                    width: 120,
                    enableButtonWrap: true,
                  ),
                ),

                Padding(
                  padding: EdgeInsets.only(top: 20),
                  child: Text(
                    'Resting Blood Pressure? (mm Hg)',
                    style: TextStyle(
                        fontWeight: FontWeight.bold,
                        fontSize: 15
                    ),
                  ),
                ),

                Container(
                  margin: EdgeInsets.only(left: 20, right: 20, top: 5),
                  height: 48,
                  child: TextFormField(
                    validator: (value) {
                      print(value);
                      if (value.isEmpty) {
                        return 'This field cannot be empty';
                      }
                      return null;
                    },
                    keyboardType: TextInputType.number,
                    textAlign: TextAlign.center,
                    decoration: InputDecoration(
                      enabledBorder: OutlineInputBorder(
                          borderRadius: BorderRadius.all(Radius.circular(10)),
                          borderSide: BorderSide(color: Colors.grey)
                      ),
                      focusedBorder: OutlineInputBorder(
                          borderRadius: BorderRadius.all(Radius.circular(10)),
                          borderSide: BorderSide(color: Colors.grey)
                      ),
                      filled: true,
                      fillColor: Colors.white,
                    ),
                    onChanged: (value) {
                      restingBloodPressure = double.parse(value);
                    },
                  ),

                ),

                Padding(
                  padding: EdgeInsets.only(top: 20),
                  child: Text(
                    'Cholesterol? (mm/dl)',
                    style: TextStyle(
                        fontWeight: FontWeight.bold,
                        fontSize: 15
                    ),
                  ),
                ),

                Container(
                  margin: EdgeInsets.only(left: 20, right: 20, top: 5),
                  height: 48,
                  child: TextFormField(
                    validator: (value) {
                      print(value);
                      if (value.isEmpty) {
                        return 'This field cannot be empty';
                      }
                      return null;
                    },
                    keyboardType: TextInputType.number,
                    textAlign: TextAlign.center,
                    decoration: InputDecoration(
                      enabledBorder: OutlineInputBorder(
                          borderRadius: BorderRadius.all(Radius.circular(10)),
                          borderSide: BorderSide(color: Colors.grey)
                      ),
                      focusedBorder: OutlineInputBorder(
                          borderRadius: BorderRadius.all(Radius.circular(10)),
                          borderSide: BorderSide(color: Colors.grey)
                      ),
                      filled: true,
                      fillColor: Colors.white,
                    ),
                    onChanged: (value) {
                      cholesterol = double.parse(value);
                    },
                  ),

                ),

                Padding(
                  padding: EdgeInsets.only(top: 20),
                  child: Text(
                    'Presence of Fasting Blood Sugar?',
                    style: TextStyle(
                        fontWeight: FontWeight.bold,
                        fontSize: 15
                    ),
                  ),
                ),

                Padding(
                  padding: EdgeInsets.only(top: 2),
                  child: CustomRadioButton(
                    elevation: 10,
                    absoluteZeroSpacing: false,
                    unSelectedColor: Colors.white,
                    buttonLables: [
                      'Yes',
                      'No',
                    ],
                    buttonValues: [
                      1,
                      0,
                    ],
                    buttonTextStyle: ButtonTextStyle(
                        selectedColor: Colors.white,
                        unSelectedColor: Colors.black,
                        textStyle: TextStyle(fontSize: 12, fontWeight: FontWeight.bold)),
                    radioButtonValue: (value) {
                      setState(() {
                        fbs = value;
                      });
                      print(value);
                    },
                    selectedColor: Colors.green[700],
                    width: 70,
                    enableButtonWrap: true,
                  ),
                ),

                Padding(
                  padding: EdgeInsets.only(top: 20),
                  child: Text(
                    'Resting ECG',
                    style: TextStyle(
                        fontWeight: FontWeight.bold,
                        fontSize: 15
                    ),
                  ),
                ),

                Padding(
                  padding: EdgeInsets.only(top: 2),
                  child: CustomRadioButton(
                    elevation: 10,
                    absoluteZeroSpacing: false,
                    unSelectedColor: Colors.white,
                    buttonLables: [
                      'Normal',
                      'Having ST-T',
                      'Hypertrophy'
                    ],
                    buttonValues: [
                      0,
                      1,
                      2,
                    ],
                    buttonTextStyle: ButtonTextStyle(
                        selectedColor: Colors.white,
                        unSelectedColor: Colors.black,
                        textStyle: TextStyle(fontSize: 12, fontWeight: FontWeight.bold)),
                    radioButtonValue: (value) {
                      setState(() {
                        rer = value;
                      });
                      print(value);
                    },
                    selectedColor: Colors.green[700],
                    width: 110,
                    enableButtonWrap: true,
                  ),
                ),

                Padding(
                  padding: EdgeInsets.only(top: 20),
                  child: Text(
                    'Maximum Heart Rate Achieved?',
                    style: TextStyle(
                        fontWeight: FontWeight.bold,
                        fontSize: 15
                    ),
                  ),
                ),

                Container(
                  margin: EdgeInsets.only(left: 20, right: 20, top: 5),
                  height: 48,
                  child: TextFormField(
                    validator: (value) {
                      print(value);
                      if (value.isEmpty) {
                        return 'This field cannot be empty';
                      }
                      return null;
                    },
                    keyboardType: TextInputType.number,
                    textAlign: TextAlign.center,
                    decoration: InputDecoration(
                      enabledBorder: OutlineInputBorder(
                          borderRadius: BorderRadius.all(Radius.circular(10)),
                          borderSide: BorderSide(color: Colors.grey)
                      ),
                      focusedBorder: OutlineInputBorder(
                          borderRadius: BorderRadius.all(Radius.circular(10)),
                          borderSide: BorderSide(color: Colors.grey)
                      ),
                      filled: true,
                      fillColor: Colors.white,
                    ),
                    onChanged: (value) {
                      maxHeartRate = double.parse(value);
                    },
                  ),

                ),

                Padding(
                  padding: EdgeInsets.only(top: 20),
                  child: Text(
                    'Oldpeak - ST depression induced by exercise relative to rest (Example: 1/3.1/2.6)',
                    style: TextStyle(
                        fontWeight: FontWeight.bold,
                        fontSize: 15
                    ),
                  ),
                ),

                Container(
                  margin: EdgeInsets.only(left: 20, right: 20, top: 5),
                  height: 48,
                  child: TextFormField(
                    validator: (value) {
                      print(value);
                      if (value.isEmpty) {
                        return 'This field cannot be empty';
                      }
                      return null;
                    },
                    keyboardType: TextInputType.number,
                    textAlign: TextAlign.center,
                    decoration: InputDecoration(
                      enabledBorder: OutlineInputBorder(
                          borderRadius: BorderRadius.all(Radius.circular(10)),
                          borderSide: BorderSide(color: Colors.grey)
                      ),
                      focusedBorder: OutlineInputBorder(
                          borderRadius: BorderRadius.all(Radius.circular(10)),
                          borderSide: BorderSide(color: Colors.grey)
                      ),
                      filled: true,
                      fillColor: Colors.white,
                    ),
                    onChanged: (value) {
                      oldpeak = double.parse(value);
                    },
                  ),

                ),

                Padding(
                  padding: EdgeInsets.only(top: 20),
                  child: Text(
                    'Slope - the slope of the peak exercise ST segment',
                    style: TextStyle(
                        fontWeight: FontWeight.bold,
                        fontSize: 15
                    ),
                  ),
                ),

                Padding(
                  padding: EdgeInsets.only(top: 2),
                  child: CustomRadioButton(
                    elevation: 10,
                    absoluteZeroSpacing: false,
                    unSelectedColor: Colors.white,
                    buttonLables: [
                      'Upsloping',
                      'Flat',
                      'Downsloping'
                    ],
                    buttonValues: [
                      0,
                      1,
                      2,
                    ],
                    buttonTextStyle: ButtonTextStyle(
                        selectedColor: Colors.white,
                        unSelectedColor: Colors.black,
                        textStyle: TextStyle(fontSize: 12, fontWeight: FontWeight.bold)),
                    radioButtonValue: (value) {
                      setState(() {
                        slope = value;
                      });
                      print(value);
                    },
                    selectedColor: Colors.green[700],
                    width: 110,
                    enableButtonWrap: true,
                  ),
                ),

                Padding(
                  padding: EdgeInsets.only(top: 20),
                  child: Text(
                    'Number of major vessels colored by flouroscopy',
                    style: TextStyle(
                        fontWeight: FontWeight.bold,
                        fontSize: 15
                    ),
                  ),
                ),

                Padding(
                  padding: EdgeInsets.only(top: 2),
                  child: CustomRadioButton(
                    elevation: 10,
                    absoluteZeroSpacing: false,
                    unSelectedColor: Colors.white,
                    buttonLables: [
                      '0',
                      '1',
                      '2',
                      '3',
                    ],
                    buttonValues: [
                      0,
                      1,
                      2,
                      3,
                    ],
                    buttonTextStyle: ButtonTextStyle(
                        selectedColor: Colors.white,
                        unSelectedColor: Colors.black,
                        textStyle: TextStyle(fontSize: 12, fontWeight: FontWeight.bold)),
                    radioButtonValue: (value) {
                      setState(() {
                        noOfVessels = value;
                      });
                      print(value);
                    },
                    selectedColor: Colors.green[700],
                    width: 70,
                    enableButtonWrap: true,
                  ),
                ),

                Padding(
                  padding: EdgeInsets.only(top: 20),
                  child: Text(
                    'Thal?',
                    style: TextStyle(
                        fontWeight: FontWeight.bold,
                        fontSize: 15
                    ),
                  ),
                ),

                Padding(
                  padding: EdgeInsets.only(top: 2),
                  child: CustomRadioButton(
                    elevation: 10,
                    absoluteZeroSpacing: false,
                    unSelectedColor: Colors.white,
                    buttonLables: [
                      'Normal',
                      'Fixed Defect',
                      'Reversible Defect',
                    ],
                    buttonValues: [
                      1,
                      2,
                      3,
                    ],
                    buttonTextStyle: ButtonTextStyle(
                        selectedColor: Colors.white,
                        unSelectedColor: Colors.black,
                        textStyle: TextStyle(fontSize: 12, fontWeight: FontWeight.bold)),
                    radioButtonValue: (value) {
                      setState(() {
                        thal = value;
                      });
                      print(value);
                    },
                    selectedColor: Colors.green[700],
                    width: 140,
                    enableButtonWrap: true,
                  ),
                ),



                Container(
                  margin: EdgeInsets.only(left: 10, right: 10, top: 20, bottom: 50),
                  decoration: BoxDecoration(
                      borderRadius: BorderRadius.circular(10),
                      color: Colors.indigo[300]
                  ),
                  width: MediaQuery
                      .of(context)
                      .size
                      .width,
                  child: FlatButton(
                    color: Colors.indigo[300],
                    onPressed: () async {
                      setState(() {
                        input = [age, gender.toDouble(), chestPain.toDouble(), restingBloodPressure, cholesterol, fbs.toDouble(), rer.toDouble(),
                                  maxHeartRate, exang.toDouble(), oldpeak, slope.toDouble(), noOfVessels.toDouble(), thal.toDouble()];
                      });
                      print(input);

                      _prediction = await _model.getPrediction(input, [1, 13], DType.float32);
                      print(_prediction);
                      if (_prediction[0] > _prediction[1]) {
                        print("No Heart Disease");
                        result = false;
                      } else if (_prediction[1] > _prediction[0]) {
                        print("Heart Disease Present");
                        result = true;
                      } else {
                        result = false;
                      }

                      EasyLoading.instance
                        ..maskColor = Colors.indigo[300].withOpacity(0.5);

                      _timer?.cancel();
                      await EasyLoading.show(
                        status: 'Predicting...',
                        maskType: EasyLoadingMaskType.custom,

                      );

                      Timer(Duration(seconds: 3), () async {
                        _timer?.cancel();
                        await EasyLoading.dismiss();
                        showAlertDialog(input);
                      });
                    },
                    child: Text(
                      'Predict',
                      style: TextStyle(
                          color: Colors.white,
                          fontSize: 20,
                          letterSpacing: 2.0
                      ),
                    ),

                  ),
                ),

              ],
            ),
          )
      ),
    );
  }

  showAlertDialog(input) async {

    Widget continueButton = FlatButton(
      child: Container(
        color: Colors.blueGrey[400],
        alignment: Alignment.center,
        padding: EdgeInsets.all(10.0),
        child: Text(
          "EXIT",
          style: TextStyle(
              color: Colors.white,
              fontWeight: FontWeight.bold,
              letterSpacing: 3.0
          ),
        ),
      ),
      onPressed:  () {
        Navigator.pushAndRemoveUntil(
          context,
          MaterialPageRoute(builder: (context) => MyHomePage()),
              (Route<dynamic> route) => false,
        );
      },
    );

    // set up the AlertDialog
    AlertDialog alert = AlertDialog(
      elevation: 24.0,
      // backgroundColor: Colors.green,
      title: Center(
        child: Text(
          "HEART DISEASE TEST RESULT",
          style: TextStyle(
              fontSize: 20,
              fontStyle: FontStyle.normal,
              color: Colors.blueGrey
          ),
        ),
      ),
      content: coronaResult(),

      actions: [
        continueButton,
      ],
    );

    // show the dialog
    showDialog(
      context: context,
      barrierDismissible: false,
      barrierColor: Colors.indigo[300].withOpacity(0.5),
      builder: (BuildContext context) {
        return alert;
      },
    );
  }

  coronaResult() {
    if (result == true) {
      return Column(
        mainAxisSize: MainAxisSize.min,
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Align(
            alignment: Alignment.center,
            child: Padding(
              padding: EdgeInsets.only(left: 0, top: 0),
              child: Image(
                image: new AssetImage(
                  "assets/notsafe.png",
                ),
                fit: BoxFit.fitWidth,
                height: 50,
                width: 50,
              ),
            ),
          ),

          Align(
            alignment: Alignment.center,
            child: Padding(
                padding: EdgeInsets.only(left: 0, top: 15),
                child: Text(
                  'HEART DISEASE DETECTED!',
                  style: TextStyle(
                      fontSize: 18,
                      fontWeight: FontWeight.bold,
                      color: Colors.red[700],
                      letterSpacing: 2.0
                  ),
                )
            ),
          ),

          Align(
            alignment: Alignment.center,
            child: Padding(
                padding: EdgeInsets.only(left: 0, top: 15),
                child: Center(
                  child: Text(
                    'Please contact a doctor!',
                    style: TextStyle(
                        fontSize: 12,
                        fontWeight: FontWeight.bold,
                        color: Colors.red[700]
                    ),
                  ),
                )
            ),
          ),
        ],
      );
    } else {
      return Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Align(
            alignment: Alignment.center,
            child: Padding(
              padding: EdgeInsets.only(left: 0, top: 0),
              child: Image(
                image: new AssetImage(
                  "assets/safe.png",
                ),
                fit: BoxFit.fitWidth,
                height: 50,
                width: 50,
              ),
            ),
          ),

          Align(
            alignment: Alignment.center,
            child: Padding(
                padding: EdgeInsets.only(left: 0, top: 15),
                child: Text(
                  'NO HEART DISEASE!',
                  style: TextStyle(
                      fontSize: 18,
                      fontWeight: FontWeight.bold,
                      color: Colors.green[600],
                      letterSpacing: 2.0
                  ),
                )
            ),
          ),

          Align(
            alignment: Alignment.center,
            child: Padding(
                padding: EdgeInsets.only(left: 0, top: 15),
                child: Text(
                  'No need to worry you are safe!',
                  style: TextStyle(
                      fontSize: 12,
                      fontWeight: FontWeight.bold,
                      color: Colors.green[600]
                  ),
                )
            ),
          ),
        ],
      );
    }
  }

}