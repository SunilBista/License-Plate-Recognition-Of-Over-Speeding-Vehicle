<!DOCTYPE html>
<html lang="en">
<head>
    <link rel="icon" href="img/4.png" type="image/x-icon" />
    <title>Traffic Speed Monitor</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.4.0/css/bootstrap.min.css">
    <link rel="stylesheet" href="style.css">
    <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.4.1/jquery.min.js"></script>
    <script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.4.0/js/bootstrap.min.js"></script>
    <script>
    function detail(x)
    {
        $("#image").attr({src:'final project/vehicle/'+x+'.png'})
    }
    </script>
  
</head>



    
<body style="background-image:url(img/5.jpg)">
<div class="container">
<div class="page-header">
<h1 class="text-center" style="color:yellow"><img src="img/4.png" height="5%" width="5%" class="img-rounded"><b> VEHICLE SPEED LIMIT MONITOR</b></h1>
</div>
<div class="row">
<div class="col-md-8">
<div class="jumbotron" style="opacity:0.9" >
    <h3 style="color:blue"><b>Vehicle's details and description:</b></h3>
    <table class="table table-bordered table-striped table-hover">
        <thead>
            <tr>
                <th>Car ID</th>
                <th>Speed</th>
                <th>Description</th>
                <th>Liscence Plate Number</th>
            </tr>
        </thead>
    </table>
</div>
</div>

<div class="col-md-4">
<img id = "image" src="final project/vehicle/t.png" width="100%"/> 
</div>
</div>

        <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.4.1/jquery.min.js"></script>
        <script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.6/js/bootstrap.min.js"></script>
        <script>
            $.getJSON("final project/data.json", function(data){
                var items= [];
                $.each(data, function(key,val){
                    // items.push("<input type='button' class='btn btn-success'  onclick='detail("+key+")' value='Car "+val.Car_ID+"'/><br /><br />")
                    items.push("<tr>");
                    items.push("<td id='' "+key+"''>"+"<input type='button' class='btn btn-success'  onclick='detail("+val.Car_ID+")' value='Car "+val.Car_ID+"'/></td>");
                    items.push("<td id='' "+key+"''>" + val.Speed + "</td>");
                    items.push("<td id='' "+key+"''>" + val.Description + "</td>");
                    items.push("<td id='' "+key+"''>" + val.License_plate_number  + "</td>");
                    items.push("</tr>");
                    });
                // items.push("</div>")
                // let s = items.join("")
                // console.log(s)
                // $("#changeable").html(s)
                 $('<tbody/>', {html: items.join("")}).appendTo("table");  
            });


        </script>
      
    </div>
    </div>

</body>
</html>