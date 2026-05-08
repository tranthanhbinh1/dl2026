@main def gradientDescent(): Unit = 
    def f(x: Float): Float = 
        return x

    def f_(x: Float): Float = 
        return 2 * x

    def gd(x0: Float, lr: Float, thres: Float) = 
        var x: Float = x0

        while math.abs(x) >= thres do
            val nextX: Float = x - lr * f_(x)

            val message = f"x: $x%.6f - f(x): ${f(x)}%.6f - next x: $nextX%.6f"
            println(message)

            x = nextX


    val x: Float = 10.0f
    val lr: Float = 0.1f
    val thres: Float = 0.001f

    gd(x, lr, thres)

  
