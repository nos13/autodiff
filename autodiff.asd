(asdf:defsystem #:autodiff
  :serial t
  :description "automatic differentiation library for gradient & hessian"
  :author "nos"
  :license "NYSL 0.9982"
  :version "0.0.0"
  :depends-on (#:cl-generic-arithmetic
	       #:alexandria)
  :components ((:file "package")
               (:file "autodiff")))








