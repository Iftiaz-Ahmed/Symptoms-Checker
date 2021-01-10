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

class CovidChecker extends StatefulWidget{
  CovidChecker({Key key}) : super(key: key);

  @override
  _CovidCheckerState createState() => _CovidCheckerState();
}

class _CovidCheckerState extends State<CovidChecker> {
  int breathing = 0;
  int fever = 0;
  int dryCough = 0;
  int soreThroat = 0;
  int runningNose = 0;
  int asthma = 0;
  int lungDisease = 0;
  int headache = 0;
  int heartDisease = 0;
  int diabetes = 0;
  int hyperTension = 0;
  int fatigue = 0;
  int gastric = 0;
  int abroadTravel = 0;
  int closeContact = 0;
  int attendedLargeGathering = 0;
  int visitedPublicPlace = 0;
  int familyWorkPublicPlace = 0;
  int wearMask = 0;
  int sanitizeHands = 0;
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
    String _modelPath = "assets/models/covidTest.pt";
    try {
      _model = await PyTorchMobile.loadModel(_modelPath);
    } on PlatformException {
      print("only supported for android so far");
    }
  }

  //run a custom model with number inputs
  Future getPrediction() async {
    _prediction = await _model
        .getPrediction([1, 2, 3, 4], [1, 2, 2], DType.float32);

    setState(() {});
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
                  'COVID CHECKER',
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
                    'Breathing Problem?',
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
                      breathing = value;
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
                  'Fever?',
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
                      fever = value;
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
                  'Dry Cough?',
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
                      dryCough = value;
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
                  'Sore Throat?',
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
                      soreThroat = value;
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
                  'Running Nose?',
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
                      runningNose = value;
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
                  'Asthma?',
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
                      asthma = value;
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
                  'Chronic Lung Disease?',
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
                      lungDisease = value;
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
                  'Headache?',
                  style: TextStyle(
                      fontWeight: FontWeight.bold,
                      fontSize: 15,
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
                      headache = value;
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
                  'Heart Disease?',
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
                      heartDisease = value;
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
                  'Diabetes?',
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
                      diabetes = value;
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
                  'Hyper Tension?',
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
                      hyperTension = value;
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
                  'Fatigue?',
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
                      fatigue = value;
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
                  'Gastric Problem?',
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
                      gastric = value;
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
                  'Did you travel abroad in this pandemic?',
                  style: TextStyle(
                      fontWeight: FontWeight.bold,
                      fontSize: 13
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
                      abroadTravel = value;
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
                  'Did you come in close contact with a COVID patient?',
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
                      closeContact = value;
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
                  'Attended Large Gatherings?',
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
                      attendedLargeGathering = value;
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
                  'Visited Public Places?',
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
                      visitedPublicPlace = value;
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
                  'Does anyone in your family go out for work?',
                  style: TextStyle(
                      fontWeight: FontWeight.bold,
                      fontSize: 14
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
                      familyWorkPublicPlace = value;
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
                  'Do you weak mask?',
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
                      wearMask = value;
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
                  'Do you sanitize your hands?',
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
                      sanitizeHands = value;
                    });
                    print(value);
                  },
                  selectedColor: Colors.green[700],
                  width: 70,
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
                      input = [breathing.toDouble(),fever.toDouble(),dryCough.toDouble(),soreThroat.toDouble(),runningNose.toDouble(),asthma.toDouble(),lungDisease.toDouble(),headache.toDouble(),heartDisease.toDouble(),
                                diabetes.toDouble(),hyperTension.toDouble(),fatigue.toDouble(),gastric.toDouble(),abroadTravel.toDouble(),closeContact.toDouble(),attendedLargeGathering.toDouble(),
                                visitedPublicPlace.toDouble(),familyWorkPublicPlace.toDouble(),wearMask.toDouble(),sanitizeHands.toDouble()];
                    });
                    print(input);

                    _prediction = await _model.getPrediction(input, [1, 20], DType.float32);
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
          "COVID TEST RESULT",
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
                  'COVID POSITIVE',
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
                  'COVID NEGATIVE',
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