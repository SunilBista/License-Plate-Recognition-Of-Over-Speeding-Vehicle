<!DOCTYPE html>
<html lang="en">

<head>
    <link rel="icon" href="img/new.png" type="image/x-icon" />
    <title>Traffic Speed Monitor</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.4.0/css/bootstrap.min.css">
    <link rel="stylesheet" href="style.css">
    <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.4.1/jquery.min.js"></script>
    <script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.4.0/js/bootstrap.min.js"></script>

</head>
<body style="background-image:url(img/back.jpg)">
<?php require_once 'login.php'?>
<?php
 if (isset($_SESSION['message'])):?>
 <div class="alert alert-<?=$_SESSION['msg_type']?>">
 <?php
 echo $_SESSION['message'];
 unset($_SESSION['message']);
 ?>
 </div>
 <?php endif ?>
    <div class="container">
        <div class="page-header">
            <h1 id="ab" class="text-center" style="color:blue"><b>TRAFFIC MONITOR </b></h1>
        </div>
        <div class="col-lg-4"></div>
        <div class="col-lg-4">
            <div class="jumbotron" style="opacity:0.8">
        <h4 style="color:green" class="text-center"><label>ADMINISTRATOR</label></h4>
    <form action="login.php" method="POST">
        <div class="form-group">
            <label>Username:</label><input type="text" class="form-control" placeholder="Enter username..." name="name">
        </div>
        <div class="form-group">
<label>Password:</label><input type="password" class="form-control" placeholder="Enter password..." name="password">
</div>
<input type="submit" value="Log-in" name="submit" class="btn btn-primary">
    </form>
</div>
<div class="col-lg-4"></div>
<div>
</body>
</html>