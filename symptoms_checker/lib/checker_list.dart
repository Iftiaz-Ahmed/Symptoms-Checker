import 'dart:async';
import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';
import 'package:flutter_easyloading/flutter_easyloading.dart';
import 'package:symptoms_checker/covidChecker.dart';
import 'package:symptoms_checker/diabetesChecker.dart';
import 'package:symptoms_checker/heartDiseaseChecker.dart';





class CheckerList extends StatefulWidget{
  CheckerList({Key key}) : super(key: key);

  @override
  _CheckerListState createState() => _CheckerListState();

}

class _CheckerListState extends State<CheckerList> {


  @override
  Widget build(BuildContext context) {
    return Scaffold(
        appBar: AppBar(
          title: Text('Symptoms Checker'),
          backgroundColor: Colors.indigo[400],
          centerTitle: true,
        ),
        body: Container(
          width: MediaQuery.of(context).size.width,
          child: Stack(
            children: [
              Container(
                margin: EdgeInsets.only(top: 80),
                child: InkWell(
                  onTap: () async {
                    Navigator.push(
                      context,
                      MaterialPageRoute(builder: (context) => CovidChecker()),
                    );
                  },
                  child: Container(
                      margin: EdgeInsets.all(40.0),
                      height: MediaQuery.of(context).size.height/10,
                      decoration: BoxDecoration(
                        gradient: LinearGradient(
                            begin: Alignment.topLeft,
                            end: Alignment.bottomRight,
                            colors: [
                              Colors.indigo[500],
                              Colors.indigo[300]
                            ]
                        ),
                        boxShadow: [ //background color of box
                          BoxShadow(
                            color: Colors.indigo[200],
                            blurRadius: 5.0, // soften the shadow
                            spreadRadius: 2.0, //extend the shadow
                            offset: Offset(
                              0.0, // Move to right 10  horizontally
                              0.0, // Move to bottom 10 Vertically
                            ),
                          )
                        ],
                        borderRadius: BorderRadius.circular(10.0),
                      ),
                      child: Center(
                        child: Text(
                          'COVID Checker',
                          style: TextStyle(
                              fontSize: 15,
                              fontWeight: FontWeight.bold,
                              color: Colors.white
                          ),
                        ),
                      )
                  ),
                ),
              ),

              Container(
                margin: EdgeInsets.only(top: 190),
                child: InkWell(
                  onTap: () async {
                    Navigator.push(
                      context,
                      MaterialPageRoute(builder: (context) => DiabetesChecker()),
                    );
                  },
                  child: Container(
                      margin: EdgeInsets.all(40.0),
                      height: MediaQuery.of(context).size.height/10,
                      decoration: BoxDecoration(
                        gradient: LinearGradient(
                            begin: Alignment.topLeft,
                            end: Alignment.bottomRight,
                            colors: [
                              Colors.indigo[500],
                              Colors.indigo[300]
                            ]
                        ),
                        boxShadow: [ //background color of box
                          BoxShadow(
                            color: Colors.indigo[200],
                            blurRadius: 5.0, // soften the shadow
                            spreadRadius: 2.0, //extend the shadow
                            offset: Offset(
                              0.0, // Move to right 10  horizontally
                              0.0, // Move to bottom 10 Vertically
                            ),
                          )
                        ],
                        borderRadius: BorderRadius.circular(10.0),
                      ),
                      child: Center(
                        child: Text(
                          'Diabetes Checker',
                          style: TextStyle(
                              fontSize: 15,
                              fontWeight: FontWeight.bold,
                              color: Colors.white
                          ),
                        ),
                      )
                  ),
                ),
              ),

              Container(
                margin: EdgeInsets.only(top: 300),
                child: InkWell(
                  onTap: () async {
                    Navigator.push(
                      context,
                      MaterialPageRoute(builder: (context) => HeartDiseaseChecker()),
                    );
                  },
                  child: Container(
                      margin: EdgeInsets.all(40.0),
                      height: MediaQuery.of(context).size.height/10,
                      decoration: BoxDecoration(
                        gradient: LinearGradient(
                            begin: Alignment.topLeft,
                            end: Alignment.bottomRight,
                            colors: [
                              Colors.indigo[500],
                              Colors.indigo[300]
                            ]
                        ),
                        boxShadow: [ //background color of box
                          BoxShadow(
                            color: Colors.indigo[200],
                            blurRadius: 5.0, // soften the shadow
                            spreadRadius: 2.0, //extend the shadow
                            offset: Offset(
                              0.0, // Move to right 10  horizontally
                              0.0, // Move to bottom 10 Vertically
                            ),
                          )
                        ],
                        borderRadius: BorderRadius.circular(10.0),
                      ),
                      child: Center(
                        child: Text(
                          'Heart Disease Checker',
                          style: TextStyle(
                              fontSize: 15,
                              fontWeight: FontWeight.bold,
                              color: Colors.white
                          ),
                        ),
                      )
                  ),
                ),
              )
            ],
          ),
        )
    );
  }
}