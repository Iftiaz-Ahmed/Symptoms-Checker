import 'dart:async';

import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_easyloading/flutter_easyloading.dart';
import 'package:pytorch_mobile/enums/dtype.dart';
import 'package:pytorch_mobile/model.dart';
import 'package:pytorch_mobile/pytorch_mobile.dart';
import 'package:symptoms_checker/main.dart';

class DiabetesChecker extends StatefulWidget{
  DiabetesChecker({Key key}) : super(key: key);

  @override
  _DiabetesCheckerState createState() => _DiabetesCheckerState();
}

class _DiabetesCheckerState extends State<DiabetesChecker> {
  double pregnancies = 0;
  double glucose = 0;
  double bloodPressure = 0;
  double skinThickness = 0;
  double insulin = 0;
  double bmi = 0;
  double dpf = 0;
  double age = 0;

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
    String _modelPath = "assets/models/diabetesTest.pt";
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
                    'DIABETES CHECKER',
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
                    'Pregnancies? (Ex: 2/0)',
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
                      pregnancies = double.parse(value);
                      print(pregnancies);
                    },
                  ),

                ),

                Padding(
                  padding: EdgeInsets.only(top: 20),
                  child: Text(
                    'Glucose? (Ex: 138/84)',
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
                      glucose = double.parse(value);
                      print(glucose);
                    },
                  ),

                ),

                Padding(
                  padding: EdgeInsets.only(top: 20),
                  child: Text(
                    'Blood Pressure? (Ex: 62/82)',
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
                      bloodPressure = double.parse(value);
                    },
                  ),

                ),

                Padding(
                  padding: EdgeInsets.only(top: 20),
                  child: Text(
                    'Skin Thickness? (Ex: 35/31)',
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
                      skinThickness = double.parse(value);
                    },
                  ),

                ),

                Padding(
                  padding: EdgeInsets.only(top: 20),
                  child: Text(
                    'Insulin? (Ex: 0/125)',
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
                      insulin = double.parse(value);
                    },
                  ),

                ),

                Padding(
                  padding: EdgeInsets.only(top: 20),
                  child: Text(
                    'BMI? (Ex: 33.6/38.2)',
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
                      bmi = double.parse(value);
                    },
                  ),

                ),

                Padding(
                  padding: EdgeInsets.only(top: 20),
                  child: Text(
                    'Diabetes Pedigree Function? (Ex: 0.127/0.233)',
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
                      dpf = double.parse(value);
                    },
                  ),

                ),

                Padding(
                  padding: EdgeInsets.only(top: 20),
                  child: Text(
                    'Age? (Ex: 47/23)',
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
                    },
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
                        input = [pregnancies, glucose, bloodPressure, skinThickness, insulin, bmi, dpf, age];
                      });
                      print(input);

                      _prediction = await _model.getPrediction(input, [1, 8], DType.float32);
                      print(_prediction);
                      if (_prediction[0] > _prediction[1]) {
                        print("Covid Negative");
                        result = false;
                      } else if (_prediction[1] > _prediction[0]) {
                        print("Covid Positive");
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
          "DIABETES TEST RESULT",
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
                  'DIABETES POSITIVE',
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
                  'DIABETES NEGATIVE',
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