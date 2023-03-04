import 'package:flutter/material.dart';


class FirstPage extends StatelessWidget{
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Column(
        children: [
          SizedBox(height: 100,),
          LoginButtonSender(),
          SizedBox(height: 50,),
          SignupButtonSender()
        ],
      ),
    );
  }

}

class LoginButtonSender extends StatelessWidget{
  @override
  Widget build(BuildContext context) {
    return TextButton(
      onPressed: () {  },
      child: Text("Login"),
    );
  }
}

class SignupButtonSender extends StatelessWidget{
  @override
  Widget build(BuildContext context) {
    return TextButton(
      onPressed: () {  },
      child: Text("Sign Up"),
    );
  }
}