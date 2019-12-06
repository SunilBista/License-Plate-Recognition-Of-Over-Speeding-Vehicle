<?php
session_start();
include "config.php";
if (isset($_POST['submit'])) {
$username=$_POST['name'];
$password=$_POST['password'];
if(empty($username)&&empty($password)){
   $_SESSION['message']="Invalid login Credentials !!";
   $_SESSION['msg_type']="danger";
   header("location: new.php");
}
else{
   $result=mysqli_query($db, "SELECT * from user where username='$username' and passcode='$password'") or die("failed to query database" .mysql_error());
   $row=mysqli_fetch_array($result);
   if($row['username']==$username && $row['passcode']==$password){
      header("location: project.php");
   }
      else {
         $_SESSION['message']="Invalid login Credentials !!";
         $_SESSION['msg_type']="danger";
         header("location: new.php");
      }
}
}
?>
