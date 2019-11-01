<?php
	if(isset($_GET['submit'])){
		$location = $_GET['location'];
		$bhk = $_GET['bhk'];
		$sq_ft = $_GET['sq_ft'];
		$years_old = $_GET['years_old'];
		$floor = $_GET['floor'];

		$jsonf = "'{\"data\":[$location,$bhk,$sq_ft,$years_old,$floor]}'";
		$shells = "/usr/bin/python3 -W ignore /var/www/html/house_price_predict.py $jsonf";

		$result = exec($shells);
	}
?>
<!DOCTYPE html>
<html lang='en'>
	<head>
		<meta charset="utf-8"> 
		<meta name="viewport" content="width=device-width, initial-scale=1"> 
		<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap.min.css">
		<script src="https://ajax.googleapis.com/ajax/libs/jquery/3.3.1/jquery.min.js"></script>
		<script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/js/bootstrap.min.js"></script>
	</head>
	<body>
		<div class='container'>
			<div class='row'>
				<h1>House Price Prediction</h1>
				<h3>Prediction: <mark id='pred'><?php echo $result ?></mark></h3>
				<form action='index.php' method='get'>
					<div class="form-group">
						<label for="location">Location</label>
							<select class="form-control" name="location">
								<option value=0>Bommanahalli</option>
								<option value=1>White Field</option>
							</select>
					</div> 
					<div class="form-group">
						<label for="bhk">BHK</label>
							<select class="form-control" name="bhk">
								<option value=1>1</option>
								<option value=2>2</option>
								<option value=3>3</option>
							</select>
					</div>
					<div class="form-group">
						<label for="text">Sq.ft</label>
						<input type="number" class="form-control" name="sq_ft" aria-describedby="inputGroupPrepend" required>
						<div class="invalid-feedback">
							Please input Square feet.
						</div>
					</div>
					<div class="form-group">
						<label for="years_old">Years Old</label>
							<select class="form-control" name="years_old">
								<option value=1>1</option>
								<option value=2>2</option>
								<option value=3>3</option>
								<option value=4>4</option>
								<option value=5>5</option>
								<option value=6>6</option>
								<option value=7>7</option>
								<option value=8>8</option>
								<option value=9>9</option>
								<option value=10>10</option>
							</select>
					</div>
					<div class="form-group">
						<label for="floor">Floor</label>
							<select class="form-control" name="floor">
								<option value=1>1</option>
								<option value=2>2</option>
								<option value=3>3</option>
								<option value=4>4</option>
								<option value=5>5</option>
								<option value=6>6</option>
								<option value=7>7</option>
								<option value=8>8</option>
								<option value=9>9</option>
								<option value=10>10</option>
								<option value=11>11</option>
								<option value=12>12</option>
								<option value=13>13</option>
								<option value=14>14</option>
								<option value=15>15</option>
							</select>
					</div>
					<button type="submit" class="btn btn-default" name='submit'>Submit</button>
				</form>
			</div>
		</div>
	</body>
</html>
