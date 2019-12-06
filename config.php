<?php
 $server="localhost";
 $username="root";
 $password="";
 $database="validation";
$db = mysqli_connect($server,$username,$password,$database);
if(!$db){
   echo "connection sucessfully not established";
}

?>