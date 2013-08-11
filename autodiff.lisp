;; library for calculating gradient and hessian for given function, with autodifferentiation
;;  function is needed to be defined after loading cl-generic-arithmetic.

;; todo: ? define-autodiffs can't cope with &optional in lambda list
;;       ? in loop macro, sum keyword doesn't work with this (as cl-generic-arithmetic doesn't)
;;       ? ad-defun is too dirty
;;       ? defgeneric is not so fast, so consider defining other approach such as cl-autodiff's
(ql:quickload "cl-generic-arithmetic")
(provide 'autodiff)
(defpackage autodiff
  (:use :cl/ga)
  (:export gradient hessian list-to-column-vector ad-defun))

(in-package autodiff)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;; 
;; forward accumulation
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(defstruct dual
  (x  0)
  (dx 0))

(defmethod var ((x number))		;for data for variate
  (make-dual :x x :dx 1))

;; example 
(defun newton-sqrt (x)
  "example from http://www.kmonos.net/wlog/123.html"
  (let ((y 1))
    (dotimes (i 10)
      (setf y (/ (+ y (/ x y)) 2)))
    y))


;; (newton-sqrt (var 2.0)) => #S(DUAL :X 1.4142135 :DX 0.35355338)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;; 
;; backward accumulation
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;; from: hessian with autodifferintiation,
;; http://uhra.herts.ac.uk/bitstream/handle/2299/4335/903836.pdf?sequence=1

(defstruct calc-node
  (op   nil)
  (x    0)
  (barx 0)
  (args nil)
  (derivs nil))

(defvar *nodes* nil)

(defun reverse-accumulate ()
  (when *nodes*
    (let ((node (pop *nodes*)))
      (loop 
	 for a in (calc-node-args   node)
	 for d in (calc-node-derivs node)
	 do (incf (calc-node-barx a) (* (calc-node-barx node) d)))
      (reverse-accumulate))))

(defun gradient (func &rest args)
  "calculate func value and gradient at args.
try (gradient * 1 2 3)"
  (let* ((*nodes* nil)
	 (initial-nodes (mapcar (lambda (a) (make-calc-node :x a)) args))
	 (result (apply func initial-nodes)))
    (setf (calc-node-barx result) (nullary-* (calc-node-barx result)))
    ;; (pprint result)
    (reverse-accumulate)
    ;; (pprint result)
    (values (mapcar #'calc-node-barx initial-nodes) 
	    (calc-node-x result))))

(defun list-to-column-vector (list)
  (let* ((len (length list))
	 (arr (make-array (list len 1))))
    (dotimes (i len)
      (setf (aref arr i 0) (nth i list)))
    arr))

(defun variate-nth (list nth)
  (when list
    (cons (if (zerop nth) (var (car list)) (car list)) (variate-nth (cdr list) (1- nth)))))

(defun get-dw (n)
  (if (numberp n)
      0
      (dual-dx n)))

(defun hessian (func &rest args)
  "calculates hessian for func at args.
return: first value: hessian in array, second value: evaluated value at args.
try (hessian (lambda (x y z) (+ (* x x) (* 2 y y) (* 3 z z) (* 2 x y) (* 2 x z) 3)) 2 3 4)"
  (let (results grads)
    (dotimes (i (length args))
      (multiple-value-bind (grad result)
	  (apply #'gradient func (variate-nth args i))
	(push result results)
	(push grad   grads)))
    (values 
     (let ((tmp (mapcar (lambda (l) (mapcar #'get-dw l)) (nreverse grads))))
       (make-array (list (length tmp) (length (car tmp)))
		   :initial-contents tmp))
     (dual-x (car results)))
    ))


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;; 
;; differentiation definitions
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;; 

(defmacro push% (obj place)
  "same as push, but returns not place but obj"
  (let ((o (gensym)))
  `(let ((,o ,obj))
     (push ,o ,place)
     ,o)))


;; '(defmethod binary-+ ((x dual)  (y dual))
;;   (make-dual :x  (binary-+ (dual-x x) (dual-x y))
;; 	     :dx (let ((pderivs (funcall (get 'binary-+ 'pderiv) (dual-x x) (dual-x y))))
;; 		   (+ (* (dual-dx x) (nth 0 pderivs))
;; 		      (* (dual-dx y) (nth 1 pderivs))))))

(defmacro define-arithmetic-dual (funcname args)
  (when (member '&optional args)
    (error "~A ~A: function with &optional not implemented" funcname args))
  `(progn
     (defmethod ,funcname ,(loop for a in args collect (list a 'dual))
       (make-dual :x (,funcname ,@(loop for a in args collect (list 'dual-x a)))
		  :dx (let ((pderivs (funcall (get ',funcname 'pderiv)
					      ,@(loop for a in args collect (list 'dual-x a)))))
			(+ ,@(loop for a in args for i from 0 collect
				  (list '* (list 'dual-dx a) (list 'nth i 'pderivs)))))))
     ,@(loop for n from 1 to (- (expt 2 (length args)) 2) collect
	    `(defmethod ,funcname ,(loop for a in args for i from 0 collect
					(if (logbitp i n) (list a 'dual) (list a 'number)))
	       (,funcname ,@(loop for a in args for i from 0 collect
				 (if (logbitp i n) a (list 'make-dual :x a))))))))

;; '(defmethod binary-+ ((a calc-node) (b calc-node))
;;   (push% (make-calc-node :op 'binary-+ :args (list a b) :derivs (list 1 1)

;; method for calc-node doesn't 
(defmacro define-arithmetic-calc-node (funcname args)
  (when (member '&optional args)
    (error "~A ~A: function with &optional not implemented" funcname args))
  `(progn
     (defmethod ,funcname ,(loop for a in args collect (list a 'calc-node))
       (push% (make-calc-node :op ',funcname :args (list ,@args)
			      :derivs (funcall (get ',funcname 'pderiv)
					       ,@(loop for a in args collect (list 'calc-node-x a)))
			      :x (,funcname ,@(loop for a in args collect (list 'calc-node-x a))) :barx 0)
	      *nodes*))
     ,@(loop for n from 1 to (- (expt 2 (length args)) 2) collect
	    `(defmethod ,funcname ,(loop for a in args for i from 0 collect
					; not limit to number as in define-arithmetic-dual, to allow dual
					(if (logbitp i n) (list a 'calc-node) a))
	       (,funcname ,@(loop for a in args for i from 0 collect
				 (if (logbitp i n) a (list 'make-calc-node :x a))))))))


;; (progn (setf (get 'binary-+  'pderiv) (lambda (x y) (list 1 1)))
;;        (define-arithmetic-dual binary-+ (x y))
;;        (define-arithmetic-calc-node binary-+ (x y)))

(defmacro define-autodiffs (funcname args derivfunc)
  `(progn (setf (get ',funcname 'pderiv) #',derivfunc)
	  (define-arithmetic-dual ,funcname ,args)
	  (define-arithmetic-calc-node ,funcname ,args)))

(define-autodiffs binary-+  (x y) (lambda (x y) (list 1 1)))
(define-autodiffs unary-+   (x)   (lambda (x)   (list 1)))
(define-autodiffs nullary-+ (x)   (lambda (x)   (list 0)))
(define-autodiffs binary--  (x y) (lambda (x y) (list 1 -1)))
(define-autodiffs unary--   (x)   (lambda (x)   (list -1)))
(define-autodiffs binary-*  (x y) (lambda (x y) (list y x)))
(define-autodiffs unary-*   (x)   (lambda (x)   (list 1)))
(define-autodiffs nullary-* (x)   (lambda (x)   (list 0)))
(define-autodiffs binary-/  (x y) (lambda (x y) (list (/ y) (- (/ x y y)))))
(define-autodiffs unary-/   (x)   (lambda (x)   (list (- (/ 1 x x)))))
(define-autodiffs nullary-* (x)   (lambda (x)   (list 0)))
(define-autodiffs 1+        (x)   (lambda (x)   (list 1)))
(define-autodiffs 1-        (x)   (lambda (x)   (list 1)))
(define-autodiffs exp       (x)   (lambda (x)   (list (exp x))))
(define-autodiffs expt      (x y) (lambda (x y) (list (* y (expt x (1- y))) (* (log x) (expt x y)))))
;; (define-autodiffs log       (x &optional base) )
(define-autodiffs sqrt      (x)   (lambda (x)   (list (/ 1/2 (sqrt x)))))
(define-autodiffs sin       (x)   (lambda (x)   (list (cos x))))
(define-autodiffs cos       (x)   (lambda (x)   (list (- (sin x)))))
(define-autodiffs tan       (x)   (lambda (x)   (list (/ 1 (cos x) (cos x)))))
(define-autodiffs asin      (x)   (lambda (x)   (list (/ (sqrt (- 1 (* x x)))))))
(define-autodiffs acos      (x)   (lambda (x)   (list (- (/ (sqrt (- 1 (* x x))))))))
;; (define-autodiffs atan      (x &optional y)   (lambda (x)   (list (/ (+ (* x x) 1)))))
(define-autodiffs sinh      (x)   (lambda (x)   (list (cosh x))))
(define-autodiffs cosh      (x)   (lambda (x)   (list (sinh x))))
(define-autodiffs tanh      (x)   (lambda (x)   (list (/ 1 (cosh x) (cosh x)))))
(define-autodiffs asinh     (x)   (lambda (x)   (list (/ 1 (sqrt (+ 1 (* x x)))))))
(define-autodiffs acosh     (x)   (lambda (x)   (list (/ 1 (sqrt (+ x 1)) (sqrt (- x 1))))))
(define-autodiffs atanh     (x)   (lambda (x)   (list (/ 1 (- 1 (* x x))))))

;; others I need
(defmethod log ((x dual) &optional base)
  (make-dual :x (log (dual-x x) (if base base (exp 1)))
	     :dx (/ (dual-dx x) (dual-x x) (log base))))


(defmethod log ((a calc-node) &optional b)
  (push% (cond ((or (numberp b) (calc-node-p b))
		(when (numberp b) (setf b (make-calc-node :x b)))
		(make-calc-node :op 'log :args (list a b) :derivs (list (/ (calc-node-x a) (log (calc-node-x b)))
									(- (/ (log (calc-node-x a)) 
									      (log (calc-node-x b))
									      (log (calc-node-x b)) 
									      (calc-node-x b))))
				:x (log (calc-node-x a) (calc-node-x b)) :barx 0))
	       ((null b)
		(make-calc-node :op 'log :args (list a) :derivs (list (/ (calc-node-x a)))
				:x (log (calc-node-x a)) :barx 0))
	       (t (error "~A is unknown type for base of log" b))) *nodes*))



;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;; 
;; examples for test
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;; 


;; for test: http://heim.ifi.uio.no/~inf3330/scripting/doc/python/SciPy/tutorial/old/node14.html
(defun sq (x) (* x x))
(defun rosen (&rest xs) 
  (let ((result 0)) 
    (loop for i from 1 below (length xs) do
	 (incf result
	       (+ (* 100 (sq (- (nth i xs) 
				(sq (nth (1- i) xs)))))
		  (sq (- 1 (nth (1- i) xs))))))
    result))

;; (hessian #'rosen 1 2 3 4 5)
;;=> #2A((402 -400 0 0 0)
;;     (-400 3802 -800 0 0)
;;     (0 -800 9402 -1200 0)
;;     (0 0 -1200 17402 -1600)
;;     (0 0 0 -1600 200))

(defun symbol-into-autodiff (s)
  "if symbol is of the current package, intern it into :autodiff"
  (if (and (symbolp s) 
	   (or (eq (symbol-package s) *package*)
	       (boundp (alexandria:ensure-symbol s :autodiff))
	       (fboundp (alexandria:ensure-symbol s :autodiff))))
      (alexandria:ensure-symbol s :autodiff)
      s))

(defun map-tree (func exp)
  (if (atom exp)
      (funcall func exp)
      (mapcar (alexandria:curry #'map-tree func) exp)))

(defmacro ad-defun (name args &body body)
  "define function in autodiff namespace, enabling autodifferentiation"
  `(progn
     (defun ,name ,args ,@body)
     (defun ,(symbol-into-autodiff name) ,(mapcar (alexandria:curry #'map-tree #'symbol-into-autodiff) args)
       ,@(mapcar (alexandria:curry #'map-tree #'symbol-into-autodiff) body))))

(defmacro alias-into-ad (&rest symbols)
  `(progn
     ,@(loop for s in symbols
	  collect (list 'define-symbol-macro (symbol-into-autodiff s) s))))


(defmethod binary-<  ((x number) (y dual))
  (< x (dual-x y)))
(defmethod binary-<  ((x dual) (y number))
  (< (dual-x x) y))
(defmethod binary-<  ((x number) (y calc-node))
  (< x (calc-node-x y)))
(defmethod binary-<  ((x calc-node) (y number))
  (< (calc-node-x x) y))
(defmethod binary-<=  ((x number) (y dual))
  (<= x (dual-x y)))
(defmethod binary-<=  ((x dual) (y number))
  (<= (dual-x x) y))
(defmethod binary-<=  ((x number) (y calc-node))
  (<= x (calc-node-x y)))
(defmethod binary-<=  ((x calc-node) (y number))
  (<= (calc-node-x x) y))
